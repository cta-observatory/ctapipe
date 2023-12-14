"""
Tool for training the EnergyRegressor
"""
import numpy as np

from ctapipe.core import Tool
from ctapipe.core.traits import Int, IntTelescopeParameter, Path
from ctapipe.io import TableLoader
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

    output_path = Path(
        default_value=None,
        allow_none=False,
        directory_ok=False,
        help=(
            "Output path for the trained reconstructor."
            " At the moment, pickle is the only supported format."
        ),
    ).tag(config=True)

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
        ("o", "output"): "TrainEnergyRegressor.output_path",
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
        self.loader = self.enter_context(
            TableLoader(
                parent=self,
            )
        )
        self.n_events.attach_subarray(self.loader.subarray)
        self.regressor = EnergyRegressor(self.loader.subarray, parent=self)

        self.cross_validate = self.enter_context(
            CrossValidator(
                parent=self, model_component=self.regressor, overwrite=self.overwrite
            )
        )
        self.rng = np.random.default_rng(self.random_seed)
        self.check_output(self.output_path)

    def start(self):
        """
        Train models per telescope type.
        """

        types = self.loader.subarray.telescope_types
        self.log.info("Inputfile: %s", self.loader.input_url)
        self.log.info("Training models for %d types", len(types))
        for tel_type in types:
            self.log.info("Loading events for %s", tel_type)
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
            self.cross_validate(tel_type, table)

            self.log.info("Performing final fit for %s", tel_type)
            self.regressor.fit(tel_type, table)
            self.log.info("done")

    def finish(self):
        """
        Write-out trained models and cross-validation results.
        """
        self.log.info("Writing output")
        self.regressor.n_jobs = None
        self.regressor.write(self.output_path, overwrite=self.overwrite)
        self.loader.close()
        self.cross_validate.close()


def main():
    TrainEnergyRegressor().run()


if __name__ == "__main__":
    main()
