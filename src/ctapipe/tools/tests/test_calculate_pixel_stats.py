#!/usr/bin/env python3
"""
Test ctapipe-calculate-pixel-statistics tool
"""

import pytest
from traitlets.config.loader import Config

from ctapipe.core import run_tool
from ctapipe.core.tool import ToolConfigurationError
from ctapipe.io import read_table
from ctapipe.tools.calculate_pixel_stats import StatisticsCalculatorTool


def test_calculate_pixel_stats_tool(tmp_path, dl1_image_file):
    """check statistics calculation from pixel-wise image data files"""

    # Create a configuration suitable for the test
    tel_id = 3
    config = Config(
        {
            "StatisticsCalculatorTool": {
                "allowed_tels": [tel_id],
                "input_column_name": "image",
                "output_column_name": "statistics",
            },
            "PixelStatisticsCalculator": {
                "stats_aggregator_type": [
                    ("id", tel_id, "PlainAggregator"),
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
        StatisticsCalculatorTool(config=config),
        argv=[
            f"--input_url={dl1_image_file}",
            f"--output_path={monitoring_file}",
            "--overwrite",
        ],
        cwd=tmp_path,
        raises=True,
    )
    # Check that the output file has been created
    assert monitoring_file.exists()
    # Check that the output file is not empty
    assert (
        read_table(
            monitoring_file,
            path=f"/dl1/monitoring/telescope/statistics/tel_{tel_id:03d}",
        )["mean"]
        is not None
    )


def test_tool_config_error(tmp_path, dl1_image_file):
    """check tool configuration error"""

    # Run the tool with the configuration and the input file
    config = Config(
        {
            "StatisticsCalculatorTool": {
                "allowed_tels": [3],
                "input_column_name": "image_charges",
                "output_column_name": "statistics",
            }
        }
    )
    # Set the output file path
    monitoring_file = tmp_path / "monitoring.dl1.h5"
    # Check if ToolConfigurationError is raised
    # when the column name of the pixel-wise image data is not correct
    with pytest.raises(ToolConfigurationError):
        run_tool(
            StatisticsCalculatorTool(config=config),
            argv=[
                f"--input_url={dl1_image_file}",
                f"--output_path={monitoring_file}",
                "--overwrite",
            ],
            cwd=tmp_path,
            raises=True,
        )
    # Check if ToolConfigurationError is raised
    # when the input and output files are the same
    with pytest.raises(ToolConfigurationError):
        run_tool(
            StatisticsCalculatorTool(),
            argv=[
                f"--input_url={dl1_image_file}",
                f"--output_path={dl1_image_file}",
                "--overwrite",
            ],
            cwd=tmp_path,
            raises=True,
        )
