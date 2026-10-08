"""
Tool for training the EnergyRegressor
"""

import numpy as np

from ctapipe.core import Tool
from ctapipe.core.traits import Int, IntTelescopeParameter
from ctapipe.exceptions import InputMissing
from ctapipe.io import TableLoader
from ctapipe.io.models import ZipModelWriter
from ctapipe.reco import CrossValidator, EnergyRegressor

from .utils import read_training_events

__all__ = [
    "TrainEnergyRegressor",
]


class TrainEnergyRegressor(Tool):
    """
    Tool to train a `~ctapipe.reco.EnergyRegressor` on dl1b/dl2 data.

    The tool first performs a cross validation to give an initial estimate
    on the quality of the estimation and then finally trains one model
    per telescope type on the full dataset.
    """

    name = "ctapipe-train-energy-regressor"
    description = __doc__

    examples = """
    ctapipe-train-energy-regressor \\
        --config train_energy_regressor.yaml \\
        --input gamma.dl2.h5 \\
        --output energy_regressor.pkl
    """

    n_events = IntTelescopeParameter(
        default_value=None,
        allow_none=True,
        help=(
            "Number of events for training the model."
            " If not given, all available events will be used."
        ),
    ).tag(config=True)

    chunk_size = Int(
        default_value=100000,
        allow_none=True,
        help="How many subarray events to load at once before training on n_events.",
    ).tag(config=True)

    random_seed = Int(
        default_value=0, help="Random seed for sampling training events."
    ).tag(config=True)

    n_jobs = Int(
        default_value=None,
        allow_none=True,
        help="Number of threads to use for the reconstruction. This overwrites the values in the config of each reconstructor.",
    ).tag(config=True)

    aliases = {
        ("i", "input"): "TableLoader.input_url",
        ("o", "output"): "ZipModelWriter.output_path",
        "n-events": "TrainEnergyRegressor.n_events",
        "chunk-size": "TrainEnergyRegressor.chunk_size",
        "n-jobs": "EnergyRegressor.n_jobs",
        "cv-output": "CrossValidator.output_path",
    }

    classes = [
        TableLoader,
        EnergyRegressor,
        CrossValidator,
    ]

    def setup(self):
        """
        Initialize components from config.
        """
        try:
            self.loader = self.enter_context(TableLoader(parent=self))
        except InputMissing:
            self.log.critical(
                "Specifying TableLoader.input_url is required (via -i, --input or a config file)."
            )
            self.exit(1)

        self.writer = self.enter_context(ZipModelWriter(parent=self))

        self.n_events.attach_subarray(self.loader.subarray)
        self.regressor = EnergyRegressor(self.loader.subarray, parent=self)

        self.cross_validate = self.enter_context(
            CrossValidator(
                parent=self, model_component=self.regressor, overwrite=self.overwrite
            )
        )
        self.rng = np.random.default_rng(self.random_seed)
        self._reconstructor_written = False

    def start(self):
        """
        Train models per telescope type.
        """

        types = sorted({str(tel) for tel in self.loader.subarray.telescope_types})

        self.log.info("Inputfile: %s", self.loader.input_url)
        self.log.info("Training models for %d types", len(types))

        for tel_type in types:
            self.log.info("Loading events for %s, tel_ids:", tel_type)
            for tel_id in self.loader.subarray.get_tel_ids(tel_type):
                self.log.info("  %3d", tel_id)

            feature_names = self.regressor.features + [
                self.regressor.target,
                "true_impact_distance",
            ]
            table = read_training_events(
                loader=self.loader,
                chunk_size=self.chunk_size,
                telescope_type=tel_type,
                reconstructor=self.regressor,
                feature_names=feature_names,
                rng=self.rng,
                log=self.log,
                n_events=self.n_events.tel[tel_type],
            )

            self.log.info("Train on %s events", len(table))
            self.cross_validate(tel_type, table, keep_subarray_events=True)

            self.log.info("Performing final fit for %s", tel_type)
            self.regressor.fit(tel_type, table)

            self.log.info("Writing model for %s", tel_type)
            self.writer.write_joblib(tel_type, self.regressor._models[tel_type])
            self.log.info("done")

    def finish(self):
        """
        Write-out trained models and cross-validation results.
        """
        self.regressor.n_jobs = None
        self.writer.write_reconstructor(self.regressor)
        self.loader.close()
        self.cross_validate.close()


def main():
    TrainEnergyRegressor().run()


if __name__ == "__main__":
    main()
