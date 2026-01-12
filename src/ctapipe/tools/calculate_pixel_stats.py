"""
Perform statistics calculation from pixel-wise image data
"""

import pathlib

import numpy as np
from astropy.table import vstack

from ctapipe.containers import EventType
from ctapipe.core import Tool
from ctapipe.core.tool import ToolConfigurationError
from ctapipe.core.traits import (
    CInt,
    Path,
    Set,
    Unicode,
    classes_with_traits,
)
from ctapipe.exceptions import InputMissing
from ctapipe.io import HDF5Merger, write_table
from ctapipe.io.hdf5dataformat import DL1_COLUMN_NAMES, DL1_PIXEL_STATISTICS_GROUP
from ctapipe.io.tableloader import TableLoader
from ctapipe.monitoring.calculator import PixelStatisticsCalculator

__all__ = ["PixelStatisticsCalculatorTool"]


class PixelStatisticsCalculatorTool(Tool):
    """
    Perform statistics calculation for pixel-wise image data
    """

    name = "ctapipe-calculate-pixel-statistics"
    description = "Perform statistics calculation for pixel-wise image data"

    examples = """
    To calculate statistics of pixel-wise image data files:

    > ctapipe-calculate-pixel-statistics --TableLoader.input_url input.dl1.h5 --output_path /path/monitoring.h5 --overwrite

    """

    allowed_tels = Set(
        trait=CInt(),
        default_value=None,
        allow_none=True,
        help=(
            "List of allowed tel_ids, others will be ignored. "
            "If None, all telescopes in the input stream will be included."
        ),
    ).tag(config=True)

    input_column_name = Unicode(
        default_value="image",
        allow_none=False,
        help="Column name of the pixel-wise image data to calculate statistics",
    ).tag(config=True)

    output_path = Path(
        help="Output filename", default_value=pathlib.Path("monitoring.h5")
    ).tag(config=True)

    aliases = {
        ("i", "input_url"): "TableLoader.input_url",
        ("o", "output_path"): "PixelStatisticsCalculatorTool.output_path",
    }

    flags = {
        "overwrite": (
            {"HDF5Merger": {"overwrite": True}},
            "Overwrite existing files",
        ),
        "append": (
            {"HDF5Merger": {"append": True}},
            "Append to existing files",
        ),
    }

    classes = [
        TableLoader,
    ] + classes_with_traits(PixelStatisticsCalculator)

    def setup(self):
        if self.output_path is None:
            self.log.critical(
                "Setting output_path is required (via -o, --output or a config file)."
            )
            self.exit(1)

        # Read the input data with the 'TableLoader'
        try:
            self.input_data = self.enter_context(
                TableLoader(
                    parent=self,
                    dl1_images=True,  # Ensure that dl1 images are read
                )
            )
        except InputMissing:
            self.log.critical(
                "Specifying TableLoader.input_url is required (via -i, --input or a config file)."
            )
            self.exit(1)

        # Copy selected tables from the input file to the output file
        self.log.info(
            "Copying selected data and metadata to output destination using the HDF5Merger component."
        )
        # Disable the copy of waveforms and images in the HDF5Merger
        self.log.debug(
            "Overwriting the default configuration of the HDF5Merger "
            "component to disable the copy of waveforms and images."
        )
        with HDF5Merger(
            parent=self,
            output_path=self.output_path,
            simulation=False,
            r0_waveforms=False,
            r1_waveforms=False,
            dl1_images=False,
            processing_statistics=False,
            merge_strategy="events-single-ob",
        ) as merger:
            merger(self.input_data.input_url)
        # Select a new subarray if the allowed_tels configuration is used
        self.subarray = (
            self.input_data.subarray
            if self.allowed_tels is None
            else self.input_data.subarray.select_subarray(self.allowed_tels)
        )
        # Initialization of the statistics calculator
        self.stats_calculator = PixelStatisticsCalculator(
            parent=self, subarray=self.subarray
        )

    def start(self):
        # Iterate over the telescope ids and calculate the statistics
        for tel_id in self.subarray.tel_ids:
            # Read the whole dl1 images for one particular telescope
            dl1_table = self.input_data.read_telescope_events(
                telescopes=[
                    tel_id,
                ],
            )
            # Check if the dl1 table is empty and skip the telescope if so
            if len(dl1_table) == 0:
                self.log.warning(
                    "No dl1 images found for telescope 'tel_id=%d'. Skipping.",
                    tel_id,
                )
                continue
            # Check if the chunk size does not exceed the table length of the input data
            if self.stats_calculator.stats_aggregators[
                self.stats_calculator.stats_aggregator_type.tel[tel_id]
            ].chunking.chunk_size > len(dl1_table):
                raise ToolConfigurationError(
                    f"Change --SizeChunking.chunk_size to decrease the chunk size "
                    f"of the aggregation to a maximum of '{len(dl1_table)}' (table length of the "
                    f"input data for telescope 'tel_id={tel_id}')."
                )
            # Check if the input column name is in the table
            if self.input_column_name not in dl1_table.colnames:
                raise ToolConfigurationError(
                    f"Column '{self.input_column_name}' not found "
                    f"in the input data for telescope 'tel_id={tel_id}'."
                )
            # Check if the dl1 data is gain selected and add an extra dimension for n_channels
            for col_name in DL1_COLUMN_NAMES:
                if col_name in dl1_table.colnames and dl1_table[col_name].ndim == 2:
                    dl1_table[col_name] = dl1_table[col_name][:, np.newaxis]
            # Perform the first pass of the statistics calculation
            aggregated_stats = self.stats_calculator.first_pass(
                table=dl1_table,
                tel_id=tel_id,
                col_name=self.input_column_name,
            )
            # Check if 'chunk_shift' is configured for overlapping chunks
            aggregator = self.stats_calculator.stats_aggregators[
                self.stats_calculator.stats_aggregator_type.tel[tel_id]
            ]
            if (
                hasattr(aggregator.chunking, "chunk_shift")
                and aggregator.chunking.chunk_shift is not None
                and aggregator.chunking.chunk_shift > 0
            ):
                # Check if there are any faulty chunks to perform a second pass over the data
                if np.any(~aggregated_stats["is_valid"].data):
                    # Perform the second pass of the statistics calculation
                    aggregated_stats_secondpass = self.stats_calculator.second_pass(
                        table=dl1_table,
                        valid_chunks=aggregated_stats["is_valid"].data,
                        tel_id=tel_id,
                        col_name=self.input_column_name,
                    )
                    # Stack the statistic values from the first and second pass
                    aggregated_stats = vstack(
                        [aggregated_stats, aggregated_stats_secondpass]
                    )
                    # Sort the stacked aggregated statistic values by starting time
                    aggregated_stats.sort(["time_start"])
                else:
                    self.log.info(
                        "No faulty chunks found for telescope 'tel_id=%d'. Skipping second pass.",
                        tel_id,
                    )
            # Construct the output table name based on the event type and the selected column name
            output_table_name = f"{EventType(dl1_table['event_type'][0]).name.lower()}_{self.input_column_name}"
            # Write the aggregated statistics and their outlier mask to the output file
            write_table(
                aggregated_stats,
                self.output_path,
                f"{DL1_PIXEL_STATISTICS_GROUP}/{output_table_name}/tel_{tel_id:03d}",
                overwrite=self.overwrite,
            )
        self.log.info(
            "DL1 monitoring data was stored in '%s' under '%s'",
            self.output_path,
            f"{DL1_PIXEL_STATISTICS_GROUP}/{output_table_name}",
        )

    def finish(self):
        self.log.info("Tool is shutting down")


def main():
    # Run the tool
    tool = PixelStatisticsCalculatorTool()
    tool.run()


if __name__ == "main":
    main()
