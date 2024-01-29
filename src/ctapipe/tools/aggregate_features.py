"""
Tool to aggregate DL1 image parameters array-event-wise.
"""
import tables
from tqdm.auto import tqdm

from ..core import Tool, ToolConfigurationError
from ..core.traits import Bool, Integer, Path, flag
from ..image import FeatureAggregator
from ..io import HDF5Merger, TableLoader, write_table

__all__ = ["AggregateFeatures"]


class AggregateFeatures(Tool):
    """
    Aggregate DL1 image parameters array-event-wise.

    This tool calculates the maximal, minimal, and mean value,
    as well as the standart deviation for any given DL1 image parameter
    for all array events given as input.
    """

    name = "ctapipe-aggregate-image-parameters"
    description = __doc__
    examples = """
    ctapipe-aggregate-image-parameters \\
        --input gamma.dl1.h5 \\
        --output gamma_incl_agg.dl1.h5
    """

    input_path = Path(
        default_value=None,
        allow_none=False,
        directory_ok=False,
        exists=True,
        help="Input file containing DL1 image parameters",
    ).tag(config=True)

    output_path = Path(
        default_value=None,
        allow_none=False,
        directory_ok=False,
        help="Output file",
    ).tag(config=True)

    chunk_size = Integer(
        default_value=100000,
        allow_none=True,
        help="How many array events to load at once for making predictions",
    ).tag(config=True)

    progress_bar = Bool(
        help="Show progress bar during processing",
        default_value=True,
    ).tag(config=True)

    aliases = {
        ("i", "input"): "AggregateFeatures.input_path",
        ("o", "output"): "AggregateFeatures.output_path",
        "chunk-size": "AggregateFeatures.chunk_size",
    }

    flags = {
        **flag(
            "progress",
            "AggregateFeatures.progress_bar",
            "show a progress bar during event processing",
            "don't show a progress bar during event processing",
        ),
        **flag(
            "dl1-parameters",
            "HDF5Merger.dl1_parameters",
            "Include dl1 parameters in output",
            "Exclude dl1 parameters in output",
        ),
        **flag(
            "dl1-images",
            "HDF5Merger.dl1_images",
            "Include dl1 images in output",
            "Exclude dl1 images in output",
        ),
        **flag(
            "true-parameters",
            "HDF5Merger.true_parameters",
            "Include true parameters in output",
            "Exclude true parameters in output",
        ),
        **flag(
            "true-images",
            "HDF5Merger.true_images",
            "Include true images in output",
            "Exclude true images in output",
        ),
        "overwrite": (
            {
                "HDF5Merger": {"overwrite": True},
                "AggregateFeatures": {"overwrite": True},
            },
            "Overwrite output file if it exists",
        ),
    }

    classes = [TableLoader, FeatureAggregator]

    def setup(self):
        """Initilize components from config."""
        self.check_output(self.output_path)
        self.log.info("Copying to output destination.")
        with HDF5Merger(self.output_path, parent=self) as merger:
            merger(self.input_path)

        self.h5file = self.enter_context(tables.open_file(self.output_path, mode="r+"))
        self.loader = self.enter_context(
            TableLoader(
                self.input_path,
                parent=self,
            )
        )
        self.aggregator = FeatureAggregator(parent=self)
        if len(self.aggregator.image_parameters) == 0:
            raise ToolConfigurationError(
                "No image parameters to aggregate are specified."
            )

    def start(self):
        """Aggregate DL1 image parameters for input tables."""
        chunk_iterator = self.loader.read_telescope_events_chunked(
            self.chunk_size,
            simulated=False,
            true_parameters=False,
        )
        bar = tqdm(
            chunk_iterator,
            desc="Aggregating parameters",
            unit=" Array Events",
            total=chunk_iterator.n_total,
            disable=not self.progress_bar,
        )
        with bar:
            for chunk, (start, stop, table) in enumerate(chunk_iterator):
                self.log.debug("Aggregating for chunk %d", chunk)
                agg_table = self.aggregator.aggregate_table(table)
                write_table(
                    agg_table,
                    self.output_path,
                    "/dl1/event/subarray/aggregated_image_parameters",
                    append=True,
                )
                bar.update(stop - start)


def main():
    AggregateFeatures().run()


if __name__ == "__main__":
    main()
