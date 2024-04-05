"""
Tool for training the DispReconstructor
"""
import astropy.units as u
import numpy as np

from ctapipe.core import Tool
from ctapipe.core.traits import Bool, Int, IntTelescopeParameter, Path
from ctapipe.io import TableLoader
from ctapipe.reco import CrossValidator, DispReconstructor
from ctapipe.reco.preprocessing import horizontal_to_telescope

from .utils import read_training_events

__all__ = [
    "TrainDispReconstructor",
]


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
        ("o", "output"): "TrainDispReconstructor.output_path",
        "n-events": "TrainDispReconstructor.n_events",
        "n-jobs": "DispReconstructor.n_jobs",
        "cv-output": "CrossValidator.output_path",
    }

    classes = [TableLoader, DispReconstructor, CrossValidator]

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
        self.models = DispReconstructor(self.loader.subarray, parent=self)

        self.cross_validate = self.enter_context(
            CrossValidator(
                parent=self, model_component=self.models, overwrite=self.overwrite
            )
        )
        self.rng = np.random.default_rng(self.random_seed)
        self.check_output(self.output_path)

    def start(self):
        """
        Train models per telescope type using a cross-validation.
        """
        types = self.loader.subarray.telescope_types
        self.log.info("Inputfile: %s", self.loader.input_url)

        self.log.info("Training models for %d types", len(types))
        for tel_type in types:
            self.log.info("Loading events for %s", tel_type)
            feature_names = self.models.features + [
                "true_energy",
                "true_impact_distance",
                "subarray_pointing_lat",
                "subarray_pointing_lon",
                "true_alt",
                "true_az",
                "hillas_fov_lat",
                "hillas_fov_lon",
                "hillas_psi",
            ]
            table = read_training_events(
                loader=self.loader,
                chunk_size=self.chunk_size,
                telescope_type=tel_type,
                reconstructor=self.models,
                feature_names=feature_names,
                rng=self.rng,
                log=self.log,
                n_events=self.n_events.tel[tel_type],
            )
            table[self.models.target] = self._get_true_disp(table)
            table = table[
                self.models.features
                + [self.models.target, "true_energy", "true_impact_distance"]
            ]

            self.log.info("Train models on %s events", len(table))
            self.cross_validate(tel_type, table)

            self.log.info("Performing final fit for %s", tel_type)
            self.models.fit(tel_type, table)
            self.log.info("done")

    def _get_true_disp(self, table):
        fov_lon, fov_lat = horizontal_to_telescope(
            alt=table["true_alt"],
            az=table["true_az"],
            pointing_alt=table["subarray_pointing_lat"],
            pointing_az=table["subarray_pointing_lon"],
        )

        # numpy's trigonometric functions need radians
        psi = table["hillas_psi"].quantity.to_value(u.rad)
        cog_lon = table["hillas_fov_lon"].quantity
        cog_lat = table["hillas_fov_lat"].quantity

        delta_lon = fov_lon - cog_lon
        delta_lat = fov_lat - cog_lat

        true_disp = np.cos(psi) * delta_lon + np.sin(psi) * delta_lat
        true_sign = np.sign(true_disp)

        if self.project_disp:
            true_norm = np.abs(true_disp)
        else:
            true_norm = np.sqrt((fov_lon - cog_lon) ** 2 + (fov_lat - cog_lat) ** 2)

        return true_norm * true_sign

    def finish(self):
        """
        Write-out trained models and cross-validation results.
        """
        self.log.info("Writing output")
        self.models.n_jobs = None
        self.models.write(self.output_path, overwrite=self.overwrite)
        self.loader.close()
        self.cross_validate.close()


def main():
    TrainDispReconstructor().run()


if __name__ == "__main__":
    main()
