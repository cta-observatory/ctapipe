#!/usr/bin/env python3
"""
Test ctapipe-calculate-pixel-statistics tool
"""

import pytest
from traitlets.config.loader import Config

from ctapipe.core import run_tool
from ctapipe.core.tool import ToolConfigurationError
from ctapipe.io import read_table
from ctapipe.io.hdf5dataformat import DL1_COLUMN_NAMES, DL1_PIXEL_STATISTICS_GROUP
from ctapipe.tools.calculate_pixel_stats import PixelStatisticsCalculatorTool
from ctapipe.tools.merge import MergeTool


def test_calculate_pixel_stats_tool(tmp_path, dl1_image_file):
    """check statistics calculation from pixel-wise image data files"""

    # Create a configuration suitable for the test
    tel_id = 3
    config = Config(
        {
            "PixelStatisticsCalculatorTool": {
                "allowed_tels": [3],
            },
            "PixelStatisticsCalculator": {
                "stats_aggregator_type": [
                    ("type", "*", "PlainAggregator"),
                ],
                "outlier_detector_list": [
                    {
                        "apply_to": "mean",
                        "name": "MedianOutlierDetector",
                        "config": {"median_range_factors": [-2.0, 2.0]},
                    }
                ],
            },
            "PlainAggregator": {
                "chunking_type": "SizeChunking",
            },
            "SizeChunking": {
                "chunk_size": 1,
            },
        }
    )
    for col_name in DL1_COLUMN_NAMES:
        # Run the tool with the configuration and the input file
        run_tool(
            PixelStatisticsCalculatorTool(config=config),
            argv=[
                f"--input_url={dl1_image_file}",
                f"--output_path={tmp_path}/subarray_{col_name}_monitoring.dl1.h5",
                f"--PixelStatisticsCalculatorTool.input_column_name={col_name}",
                "--overwrite",
            ],
            cwd=tmp_path,
            raises=True,
        )
    # Run the merge tool to combine the statistics
    # from the two files into a single monitoring file
    monitoring_file = tmp_path / "monitoring.dl1.h5"
    run_tool(
        MergeTool(),
        argv=[
            f"{tmp_path}/subarray_image_monitoring.dl1.h5",
            f"{tmp_path}/subarray_peak_time_monitoring.dl1.h5",
            f"--output={monitoring_file}",
            "--merge-strategy=monitoring-only",
        ],
        cwd=tmp_path,
        raises=True,
    )
    # Check that the output file has been created
    assert monitoring_file.exists()
    # Check if the shape of the aggregated statistic values
    # has three dimension for both merged tables
    for col_name in DL1_COLUMN_NAMES:
        assert (
            read_table(
                monitoring_file,
                path=f"{DL1_PIXEL_STATISTICS_GROUP}/subarray_{col_name}/tel_{tel_id:03d}",
            )["mean"].ndim
            == 3
        )


def test_tool_config_error(tmp_path, dl1_image_file):
    """check tool configuration error"""

    # Run the tool with the configuration and the input file
    config = Config(
        {
            "PixelStatisticsCalculatorTool": {
                "input_column_name": "image_charges",
            }
        }
    )
    # Set the output file path
    monitoring_failure_colname_file = tmp_path / "monitoring_failure_colname.dl1.h5"
    # Check if ToolConfigurationError is raised
    # when the column name of the pixel-wise image data is not correct
    with pytest.raises(
        ToolConfigurationError, match="Column 'image_charges' not found"
    ):
        run_tool(
            PixelStatisticsCalculatorTool(config=config),
            argv=[
                f"--input_url={dl1_image_file}",
                f"--output_path={monitoring_failure_colname_file}",
                "--SizeChunking.chunk_size=1",
                "--overwrite",
            ],
            cwd=tmp_path,
            raises=True,
        )
    # Check if ToolConfigurationError is raised
    # when the chunk size is larger than the number of events in the input file
    monitoring_failure_chunk_size_file = (
        tmp_path / "monitoring_failure_chunk_size.dl1.h5"
    )

    with pytest.raises(
        ToolConfigurationError, match="Change --SizeChunking.chunk_size"
    ):
        run_tool(
            PixelStatisticsCalculatorTool(),
            argv=[
                f"--input_url={dl1_image_file}",
                f"--output_path={monitoring_failure_chunk_size_file}",
                "--SizeChunking.chunk_size=2500",
            ],
            cwd=tmp_path,
            raises=True,
        )
