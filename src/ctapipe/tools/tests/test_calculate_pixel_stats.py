#!/usr/bin/env python3
"""
Test ctapipe-calculate-pixel-statistics tool
"""

import pytest
from traitlets.config.loader import Config

from ctapipe.core import run_tool
from ctapipe.core.tool import ToolConfigurationError
from ctapipe.io import TableLoader, read_table
from ctapipe.io.tests.test_astropy_helpers import assert_table_equal
from ctapipe.tools.calculate_pixel_stats import PixelStatisticsCalculatorTool


def test_calculate_pixel_stats_tool(tmp_path, dl1_image_file):
    """check statistics calculation from pixel-wise image data files"""

    # Create a configuration suitable for the test
    tel_id = 3
    config = Config(
        {
            "PixelStatisticsCalculatorTool": {
                "allowed_tels": [3],
                "input_column_name": "image",
                "output_table_name": "statistics",
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
                "chunk_size": 1,
            },
        }
    )
    # Set the output file path
    monitoring_file = tmp_path / "monitoring.dl1.h5"
    # Run the tool with the configuration and the input file
    run_tool(
        PixelStatisticsCalculatorTool(config=config),
        argv=[
            f"--input_url={dl1_image_file}",
            f"--output_path={monitoring_file}",
            "--no-r0-waveforms",
            "--no-r1-waveforms",
            "--dl1-images",
            "--overwrite",
        ],
        cwd=tmp_path,
        raises=True,
    )
    # Check that the output file has been created
    assert monitoring_file.exists()
    # Check if the shape of the aggregated statistic values has three dimension
    assert (
        read_table(
            monitoring_file,
            path=f"/dl1/monitoring/telescope/statistics/tel_{tel_id:03d}",
        )["mean"].ndim
        == 3
    )
    # Check if the HDF5Merger has merged the input file
    # with the output file correctly
    with TableLoader(dl1_image_file) as loader:
        initial_tel_events = loader.read_telescope_events(
            telescopes=[tel_id], dl1_images=True
        )

    with TableLoader(monitoring_file) as loader:
        merged_tel_events = loader.read_telescope_events(
            telescopes=[tel_id], dl1_images=True
        )

    assert_table_equal(merged_tel_events, initial_tel_events)


def test_tool_config_error(tmp_path, dl1_image_file):
    """check tool configuration error"""

    # Run the tool with the configuration and the input file
    config = Config(
        {
            "PixelStatisticsCalculatorTool": {
                "input_column_name": "image_charges",
                "output_table_name": "statistics",
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
                "--StatisticsAggregator.chunk_size=1",
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
        ToolConfigurationError, match="Change --StatisticsAggregator.chunk_size"
    ):
        run_tool(
            PixelStatisticsCalculatorTool(),
            argv=[
                f"--input_url={dl1_image_file}",
                f"--output_path={monitoring_failure_chunk_size_file}",
                "--StatisticsAggregator.chunk_size=2500",
            ],
            cwd=tmp_path,
            raises=True,
        )
