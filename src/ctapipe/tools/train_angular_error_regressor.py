"""
Tool for training the AngularErrorRegressor
"""
import numpy as np

from ctapipe.core import Tool
from ctapipe.core.traits import Int, IntTelescopeParameter, Path, Unicode
from ctapipe.io import TableLoader
from ctapipe.reco import CrossValidator, AngularErrorRegressor

__all__ = [
    "TrainAngularErrorRegressor",
]


class TrainAngularErrorRegressor(Tool):

    name = "ctapipe-train-angular-error-regressor"
    description = __doc__

    examples = """
    ctapipe-train-angular-error-regressor \\
        --config train_angular_error_regressor.yaml \\
        --input gamma.dl2.h5 \\
        --output angular_error_regressor.pkl
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

    n_events = Int(
        default_value=None,
        allow_none=True,
        help=(
            "Number of events for training the model."
            " If not given, all available events will be used."
        ),
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
        ("o", "output"): "TrainAngularErrorRegressor.output_path",
        "n-events": "TrainAngularErrorRegressor.n_events",
    }

    classes = [
        TableLoader,
        AngularErrorRegressor,
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
        self.regressor = AngularErrorRegressor(self.loader.subarray, parent=self)
        self.rng = np.random.default_rng(self.random_seed)
        self.check_output(self.output_path)

    def start(self):
        """
        Run the tool.
        """
        events = self.loader.read_subarray_events(
            dl2=True,
            simulated=True,
        )

        self.log.info("Loaded %d events", len(events))
        valid = events[self.regressor.reconstructor_prefix + "_is_valid"]
        events = events[valid]
        self.log.info("Using %d valid events", len(events))
        self.regressor.fit(events)

    def finish(self):
        """
        Save the model to disk.
        """
        self.regressor.write(self.output_path)
def main():
    TrainAngularErrorRegressor().run()


if __name__ == "__main__":
    main()
