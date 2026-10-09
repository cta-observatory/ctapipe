"""
Tool for training the DispReconstructor
"""

import numpy as np

from ctapipe.core import Tool
from ctapipe.core.traits import Bool, Int, IntTelescopeParameter
from ctapipe.exceptions import InputMissing
from ctapipe.io import TableLoader
from ctapipe.io.models import ZipModelWriter
from ctapipe.reco import CrossValidator, DispReconstructor
from ctapipe.reco.disp import compute_true_disp

from .utils import read_training_events

__all__ = ["TrainDispReconstructor"]


class TrainDispReconstructor(Tool):
    """
    Tool to train a `~ctapipe.reco.DispReconstructor` on dl1b/dl2 data.

    The tool first performs a cross validation to give an initial estimate
    on the quality of the estimation and then finally trains two models
    (estimating ``norm(disp)`` and ``sign(disp)`` respectively) per
    telescope type on the full dataset.
    """

    name = "ctapipe-train-disp-reconstructor"
    description = __doc__

    examples = """
    ctapipe-train-disp-reconstructor \\
        --config train_disp_reconstructor.yaml \\
        --input gamma.dl2.h5 \\
        --output disp_models.pkl
    """

    n_events = IntTelescopeParameter(
        default_value=None,
        allow_none=True,
        help=(
            "Number of events for training the models."
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

    project_disp = Bool(
        default_value=False,
        help=(
            "If true, ``true_disp`` is the distance between shower cog and"
            " the true source position along the reconstructed main shower axis."
            "If false, ``true_disp`` is the distance between shower cog"
            " and the true source position."
        ),
    ).tag(config=True)

    aliases = {
        ("i", "input"): "TableLoader.input_url",
        ("o", "output"): "ZipModelWriter.output_path",
        "n-events": "TrainDispReconstructor.n_events",
        "n-jobs": "DispReconstructor.n_jobs",
        "cv-output": "CrossValidator.output_path",
    }

    classes = [TableLoader, DispReconstructor, CrossValidator]

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

        self.writer = self.enter_context(
            ZipModelWriter(parent=self, overwrite=self.overwrite)
        )

        self.n_events.attach_subarray(self.loader.subarray)
        self.reconstructor = DispReconstructor(self.loader.subarray, parent=self)

        self.cross_validate = self.enter_context(
            CrossValidator(
                parent=self,
                model_component=self.reconstructor,
                overwrite=self.overwrite,
            )
        )
        self.rng = np.random.default_rng(self.random_seed)

    def start(self):
        """
        Train models per telescope type using a cross-validation.
        """
        types = sorted({str(tel) for tel in self.loader.subarray.telescope_types})
        self.log.info("Inputfile: %s", self.loader.input_url)

        self.log.info("Training models for %d types", len(types))
        feature_names = self.reconstructor.features + [
            "true_energy",
            "true_impact_distance",
            "true_alt",
            "true_az",
            "hillas_fov_lat",
            "hillas_fov_lon",
            "hillas_psi",
        ]
        optional_columns = [
            "telescope_pointing_altitude",
            "telescope_pointing_azimuth",
            "subarray_pointing_frame",
            "subarray_pointing_lat",
            "subarray_pointing_lon",
        ]

        for tel_type in types:
            self.log.info("Loading events for %s, tel_ids:", tel_type)
            for tel_id in self.loader.subarray.get_tel_ids(tel_type):
                self.log.info("  %3d", tel_id)

            table = read_training_events(
                loader=self.loader,
                chunk_size=self.chunk_size,
                telescope_type=tel_type,
                reconstructor=self.reconstructor,
                feature_names=feature_names,
                optional_columns=optional_columns,
                rng=self.rng,
                log=self.log,
                n_events=self.n_events.tel[tel_type],
            )
            table[self.reconstructor.target] = compute_true_disp(
                table, self.project_disp
            )
            table = table[
                self.reconstructor.features
                + [self.reconstructor.target, "true_energy", "true_impact_distance"]
            ]

            self.log.info("Train models on %s events", len(table))
            self.cross_validate(tel_type, table)

            self.log.info("Performing final fit for %s", tel_type)
            self.reconstructor.fit(tel_type, table)
            self.log.info("Saving model for %s", tel_type)
            self.writer.write_joblib(tel_type, self.reconstructor._models[tel_type])
            self.log.info("done")

    def finish(self):
        """
        Write-out trained models and cross-validation results.
        """
        self.log.info("Writing output")
        self.reconstructor.n_jobs = None
        self.writer.write_reconstructor(self.reconstructor)
        self.loader.close()
        self.cross_validate.close()


def main():
    TrainDispReconstructor().run()


if __name__ == "__main__":
    main()
