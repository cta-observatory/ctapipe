"""
Tool for training the DirectionUncertaintyRegressor
"""
import numpy as np

from ctapipe.core import QualityQuery, Tool
from ctapipe.core.traits import Int, Path
from ctapipe.io import TableLoader
from ctapipe.reco import CrossValidator, DirectionUncertaintyRegressor

__all__ = [
    "TrainDirectionUncertaintyRegressor",
]


class TrainDirectionUncertaintyRegressor(Tool):
    name = "ctapipe-train-direction-uncertainty-regressor"
    description = __doc__

    examples = """
    ctapipe-train-direction-uncertainty-regressor \\
        --config train_direction_uncertainty_regressor.yaml \\
        --input gamma.dl2.h5 \\
        --output direction_uncertainty_regressor.pkl
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
        ("o", "output"): "TrainDirectionUncertaintyRegressor.output_path",
        "n-events": "TrainDirectionUncertaintyRegressor.n_events",
    }

    classes = [
        TableLoader,
        DirectionUncertaintyRegressor,
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
        self.regressor = DirectionUncertaintyRegressor(
            self.loader.subarray, parent=self
        )
        self.quality_query = QualityQuery(parent=self)
        self.rng = np.random.default_rng(self.random_seed)
        self.check_output(self.output_path)

    def start(self):
        """
        Run the tool.
        """
        events = self.loader.read_subarray_events(
            dl2=True,
            simulated=True,
            dl1_aggregates=True,
        )

        self.log.info("Loaded %d events", len(events))
        valid = self.quality_query.get_table_mask(events)
        valid = valid & events[f"{self.regressor.reconstructor_prefix}_is_valid"]
        events = events[valid]
        self.log.info("Using %d valid events", len(events))
        self.regressor.fit(events)

    def finish(self):
        """
        Save the model to disk.
        """
        self.regressor.write(self.output_path, overwrite=self.overwrite)


def main():
    TrainDirectionUncertaintyRegressor().run()


if __name__ == "__main__":
    main()
