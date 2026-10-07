"""
Perform statistics calculation from pixel-wise image data
"""

import pathlib

import numpy as np
from astropy.table import vstack

from ctapipe.containers import EventType, PixelStatus
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
from ctapipe.exceptions import InputMissing
from ctapipe.io import HDF5Merger, write_table
from ctapipe.io.hdf5dataformat import (
    DL1_COLUMN_NAMES,
    DL1_PIXEL_HISTOGRAMS_GROUP,
    DL1_PIXEL_STATISTICS_GROUP,
)
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

    per_channel_statistics = Bool(
        default_value=False,
        help=(
            "If the input data is gain selected, fill each pixel value into the "
            "gain channel given by 'selected_gain_channel', so that the statistics "
            "are calculated per gain channel. Unfilled entries are set to NaN and "
            "ignored by the aggregation."
        ),
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
        "per-channel": (
            {"PixelStatisticsCalculatorTool": {"per_channel_statistics": True}},
            "Calculate per-channel statistics for gain selected input data",
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
        """Iterate over all telescopes, process their statistics, and save the results."""
        # Track the last used table group and name for the summary log statement
        output_table_name = ""
        table_group = ""

        for tel_id in self.subarray.tel_ids:
            # 1. Load and validate
            dl1_table = self.input_data.read_telescope_events(telescopes=[tel_id])
            if not self._is_valid_table(dl1_table, tel_id):
                continue

            # 2. Reshape and calculate stats
            self._reshape_dl1_dimensions(dl1_table, tel_id)
            aggregated_stats = self._process_telescope_stats(dl1_table, tel_id)

            # 3. Determine output paths and write out results
            event_type_name = EventType(dl1_table["event_type"][0]).name.lower()
            output_table_name = f"{event_type_name}_{self.input_column_name}"
            table_group = (
                DL1_PIXEL_HISTOGRAMS_GROUP
                if "histogram" in aggregated_stats.colnames
                else DL1_PIXEL_STATISTICS_GROUP
            )

            write_table(
                aggregated_stats,
                self.output_path,
                f"{table_group}/{output_table_name}/tel_{tel_id:03d}",
                overwrite=self.overwrite,
            )

        if output_table_name and table_group:
            self.log.info(
                "DL1 monitoring data was stored in '%s' under '%s'",
                self.output_path,
                f"{table_group}/{output_table_name}",
            )

    def _is_valid_table(self, table, tel_id):
        # Check if the dl1 table is empty and skip the telescope if so
        if len(table) == 0:
            self.log.warning(
                "No dl1 images found for telescope 'tel_id=%d'. Skipping.",
                tel_id,
            )
            return False
        # Check if the chunk size does not exceed the table length of the input data
        if self.stats_calculator.stats_aggregators[
            self.stats_calculator.stats_aggregator_type.tel[tel_id]
        ].chunking.chunk_size > len(table):
            raise ToolConfigurationError(
                f"Change --SizeChunking.chunk_size to decrease the chunk size "
                f"of the aggregation to a maximum of '{len(table)}' (table length of the "
                f"input data for telescope 'tel_id={tel_id}')."
            )
        # Check if the input column name is in the table
        if self.input_column_name not in table.colnames:
            raise ToolConfigurationError(
                f"Column '{self.input_column_name}' not found "
                f"in the input data for telescope 'tel_id={tel_id}'."
            )
        # Check if the gain channel information is available for per-channel statistics
        if (
            self.per_channel_statistics
            and table[self.input_column_name].ndim == 2
            and self.subarray.tel[tel_id].camera.readout.n_channels > 1
            and "pixel_status" not in table.colnames
        ):
            raise ToolConfigurationError(
                "Per-channel statistics requested, but column 'pixel_status' "
                f"not found in the gain selected input data for telescope "
                f"'tel_id={tel_id}'."
            )
        return True

    def _reshape_dl1_dimensions(self, dl1_table, tel_id):
        """
        Check if the dl1 data is gain selected and add an extra dimension.

        If ``per_channel_statistics`` is enabled, the gain selected data is filled
        into an array of shape (n_events, n_channels, n_pixels) according to
        ``selected_gain_channel``, which is derived from ``pixel_status``, with NaN
        for the not selected channel. Pixels without a stored gain channel
        (e.g. broken pixels) are NaN in both channels.
        """
        n_channels = self.subarray.tel[tel_id].camera.readout.n_channels
        fill_per_channel = (
            self.per_channel_statistics
            and n_channels > 1
            and "pixel_status" in dl1_table.colnames
        )
        gain_selected_cols = [
            col
            for col in DL1_COLUMN_NAMES
            if col in dl1_table.colnames and dl1_table[col].ndim == 2
        ]
        if not fill_per_channel:
            for col in gain_selected_cols:
                dl1_table[col] = dl1_table[col][:, np.newaxis]
            return

        # channel info: 1 = only high gain stored, 2 = only low gain stored
        channel_info = PixelStatus.get_channel_info(
            np.asarray(dl1_table["pixel_status"])
        )
        stored = (channel_info == 1) | (channel_info == 2)
        event_index, pixel_index = np.nonzero(stored)
        selected_gain_channel = channel_info[stored].astype(np.int8) - 1
        for col in gain_selected_cols:
            unit = dl1_table[col].unit
            data = np.asarray(dl1_table[col])
            per_channel_data = np.full(
                (data.shape[0], n_channels, data.shape[1]), np.nan, dtype=np.float32
            )
            per_channel_data[event_index, selected_gain_channel, pixel_index] = data[
                event_index, pixel_index
            ]
            dl1_table[col] = per_channel_data
            dl1_table[col].unit = unit

    def _process_telescope_stats(self, dl1_table, tel_id):
        """Perform first and (if necessary) second pass statistics calculation."""
        # First pass
        stats = self.stats_calculator.first_pass(
            table=dl1_table,
            tel_id=tel_id,
            col_name=self.input_column_name,
        )

        # Check if 'chunk_shift' is configured for overlapping chunks
        agg_type = self.stats_calculator.stats_aggregator_type.tel[tel_id]
        aggregator = self.stats_calculator.stats_aggregators[agg_type]
        has_chunk_shift = (getattr(aggregator.chunking, "chunk_shift") or 0) > 0

        if not has_chunk_shift:
            return stats

        # Guard clause: Skip second pass if no faulty chunks exist
        if not np.any(~stats["is_valid"].data):
            self.log.info(
                "No faulty chunks found for telescope 'tel_id=%d'. Skipping second pass.",
                tel_id,
            )
            return stats

        # Second pass execution and combining results
        stats_secondpass = self.stats_calculator.second_pass(
            table=dl1_table,
            valid_chunks=stats["is_valid"].data,
            tel_id=tel_id,
            col_name=self.input_column_name,
        )

        combined_stats = vstack([stats, stats_secondpass])
        combined_stats.sort(["time_start"])
        return combined_stats

    def finish(self):
        self.log.info("Tool is shutting down")


def main():
    # Run the tool
    tool = PixelStatisticsCalculatorTool()
    tool.run()


if __name__ == "main":
    main()
