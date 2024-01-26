
import tables
from tqdm.auto import tqdm

from ctapipe.core import Tool

from ctapipe.core.traits import Path, Integer, Bool, classes_with_traits, flag

from ctapipe.io import TableLoader, HDF5Merger, write_table
from ctapipe.reco import Reconstructor


__all__ = [
    "ApplyDirectionUncertaintyModel",
]

class ApplyDirectionUncertaintyModel(Tool):
    """
    Apply the direction uncertainty model to a DL2 file.

    Model needs to be trained with
    `~ctapipe.tools.train_direction_uncertainty_regressor.TrainDirectionUncertaintyRegressor`.
    """

    name = "ctapipe-apply-direction-uncertainty-model"
    description = __doc__

    examples = """
    ctapipe-apply-direction-uncertainty-model \\
        --input gamma.dl2.h5 \\
        --reconstructor direction_uncertainty_regressor.pkl \\
        --output gamma_applied.dl2.h5
    """

    input_url = Path(
        default_value=None,
        allow_none=False,
        directory_ok=False,
        exists=True,
        help="Input dl2 file",
    ).tag(config=True)

    output_path = Path(
        default_value=None,
        allow_none=False,
        directory_ok=False,
        help="Output file",
    ).tag(config=True)

    reconstructor_path = Path(
        default_value=None,
        allow_none=False,
        directory_ok=False,
        exists=True,
        help="Path to trained reconstructor to be applied to the input data",
    ).tag(config=True)

    chunk_size = Integer(
        default_value=100000,
        allow_none=True,
        help="How many subarray events to load at once",
    ).tag(config=True)

    n_jobs = Integer(
        default_value=None,
        allow_none=True,
        help="Number of threads to use for the reconstruction. This overwrites the values in the config",
    ).tag(config=True)

    progress_bar = Bool(
        help="show progress bar during processing",
        default_value=True,
    ).tag(config=True)

    aliases = {
        ("i", "input"): "ApplyDirectionUncertaintyModel.input_url",
        ("o", "output"): "ApplyDirectionUncertaintyModel.output_path",
        ("r", "reconstructor"): "ApplyDirectionUncertaintyModel.reconstructor_path",
        "n-jobs": "ApplyDirectionUncertaintyModel.n_jobs",
        "chunk-size": "ApplyDirectionUncertaintyModel.chunk_size",
    }

    flags = {
        **flag(
            "progress",
            "ProcessorTool.progress_bar",
            "show a progress bar during event processing",
            "don't show a progress bar during event processing",
        ),
        **flag(
            "dl1-parameters",
            "HDF5Merger.dl1_parameters",
            "Include dl1 parameters",
            "Exclude dl1 parameters",
        ),
        **flag(
            "true-parameters",
            "HDF5Merger.true_parameters",
            "Include true parameters",
            "Exclude true parameters",
        ),
        "overwrite": (
            {
                "HDF5Merger": {"overwrite": True},
                "ApplyDirectionUncertaintyModel": {"overwrite": True},
            },
            "Overwrite output file if it exists",
        ),
    }

    classes = [TableLoader] + classes_with_traits(Reconstructor)

    def setup(self):
        """
        Initialize components from config.
        """
        self.check_output(self.output_path)
        self.log.info("Copying to output destination.")
        with HDF5Merger(self.output_path, parent=self) as merger:
            merger(self.input_url)
        self.h5file = self.enter_context(tables.open_file(self.output_path, mode="r+"))
        self.loader = self.enter_context(
            TableLoader(
                self.input_url,
                parent=self,
            )
        )
        self.regressor = Reconstructor.read(self.reconstructor_path, parent=self, subarray=self.loader.subarray)
        if self.n_jobs:
            self.regressor.n_jobs = self.n_jobs

    def start(self):
        """
        Apply the model to the input file.
        """
        chunk_iterator = self.loader.read_subarray_events_chunked(
            self.chunk_size,
            simulated=False,
            observation_info=True,
            dl1_aggregates=True,
        )
        bar = tqdm(
            chunk_iterator,
            desc="Applying direction uncertainty model",
            unit="Subarray events",
            total=chunk_iterator.n_total,
            disable=not self.progress_bar,
        )
        with bar:
            for chunk, (start, stop, table) in enumerate(chunk_iterator):
                self.log.debug("Events read from chunk %d: %d", chunk, len(table))
                self._apply(self.regressor, table, start=start, stop=stop)
                self.log.debug("Events after applying direction uncertainty model: %d", len(table))
                bar.update(stop - start)

    def _apply(self, reconstructor, table, start=None, stop=None):
        prediction = reconstructor.predict_subarray_table(table)
        prefix = reconstructor.reconstructor_prefix
        new_columns = prediction.colnames
        self.log.debug("Writing to output file")
        for col in new_columns:
            table[col] = prediction[col]
        output_columns = ["obs_id", "event_id"] + new_columns
        write_table(
            table[output_columns],
            self.output_path,
            f"/dl2/event/subarray/geometry/{prefix}_uncertainty",
            append=True,
        )

def main():
    ApplyDirectionUncertaintyModel().run()


if __name__ == "__main__":
    main()