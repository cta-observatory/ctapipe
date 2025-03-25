"""
Perform statistics calculation from pixel-wise image data
"""

import pathlib

import numpy as np
from astropy.table import vstack

from ctapipe.core import Tool
from ctapipe.core.tool import ToolConfigurationError
from ctapipe.core.traits import (
    Bool,
    CInt,
    Path,
    Set,
    Unicode,
    classes_with_traits,
)
from ctapipe.instrument import SubarrayDescription
from ctapipe.io import write_table
from ctapipe.io.tableloader import TableLoader
from ctapipe.monitoring.calculator import PixelStatisticsCalculator


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

    output_table_name = Unicode(
        default_value="statistics",
        allow_none=False,
        help="Table name of the output statistics",
    ).tag(config=True)

    output_path = Path(
        help="Output filename", default_value=pathlib.Path("monitoring.h5")
    ).tag(config=True)

    overwrite = Bool(help="Overwrite output file if it exists").tag(config=True)

    aliases = {
        ("i", "input_url"): "TableLoader.input_url",
        ("o", "output_path"): "PixelStatisticsCalculatorTool.output_path",
    }

    flags = {
        "overwrite": (
            {"PixelStatisticsCalculatorTool": {"overwrite": True}},
            "Overwrite existing files",
        ),
    }

    classes = [
        TableLoader,
    ] + classes_with_traits(PixelStatisticsCalculator)

    DL1_COLUMN_NAMES = ["image", "peak_time"]

    def setup(self):
        # Read the input data with the 'TableLoader'
        self.input_data = self.enter_context(
            TableLoader(
                parent=self,
            )
        )
        # Check that the input and output files are not the same
        if self.input_data.input_url == self.output_path:
            raise ToolConfigurationError(
                "Input and output files are same. Fix your configuration / cli arguments."
            )
        if "dl1_images" in self.input_data.config.TableLoader:
            if not self.input_data.dl1_images:
                raise ToolConfigurationError(
                    "The TableLoader must read dl1 images. Set 'dl1_images' to True."
                )
        self.input_data.dl1_images = True
        # Load the subarray description from the input file
        subarray = SubarrayDescription.from_hdf(self.input_data.input_url)
        # Select a new subarray if the allowed_tels configuration is used
        self.subarray = (
            subarray
            if self.allowed_tels is None
            else subarray.select_subarray(self.allowed_tels)
        )
        # Initialization of the statistics calculator
        self.stats_calculator = PixelStatisticsCalculator(
            parent=self, subarray=self.subarray
        )

    def start(self):
        # Iterate over the telescope ids and calculate the statistics
        for tel_id in self.subarray.tel_ids:
            # Read the whole dl1 images for one particular telescope
            dl1_table = self.input_data.read_telescope_events_by_id(
                telescopes=[
                    tel_id,
                ],
            )[tel_id]
            # Check if the chunk size does not exceed the table length of the input data
            if self.stats_calculator.stats_aggregators[
                self.stats_calculator.stats_aggregator_type.tel[tel_id]
            ].chunk_size > len(dl1_table):
                raise ToolConfigurationError(
                    f"Change --StatisticsAggregator.chunk_size to decrease the chunk size "
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
            for col_name in self.DL1_COLUMN_NAMES:
                if col_name in dl1_table.colnames and dl1_table[col_name].ndim == 2:
                    dl1_table[col_name] = dl1_table[col_name][:, np.newaxis]
            # Perform the first pass of the statistics calculation
            aggregated_stats = self.stats_calculator.first_pass(
                table=dl1_table,
                tel_id=tel_id,
                col_name=self.input_column_name,
            )
            # Check if 'chunk_shift' is selected
            if self.stats_calculator.chunk_shift is not None:
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
            # Add metadata to the aggregated statistics
            aggregated_stats.meta["event_type"] = dl1_table["event_type"][0]
            aggregated_stats.meta["input_column_name"] = self.input_column_name
            # Write the aggregated statistics and their outlier mask to the output file
            write_table(
                aggregated_stats,
                self.output_path,
                f"/dl1/monitoring/telescope/{self.output_table_name}/tel_{tel_id:03d}",
                overwrite=self.overwrite,
            )
        self.log.info(
            "DL1 monitoring data was stored in '%s' under '%s'",
            self.output_path,
            f"/dl1/monitoring/telescope/{self.output_table_name}",
        )

    def finish(self):
        # Store the subarray description in the output file
        self.subarray.to_hdf(self.output_path, overwrite=self.overwrite)
        self.log.info(
            "Subarray description was stored in '%s'",
            self.output_path,
        )
        self.log.info("Tool is shutting down")


def main():
    # Run the tool
    tool = PixelStatisticsCalculatorTool()
    tool.run()


if __name__ == "main":
    main()
