#!/usr/bin/env python3
"""
Test ctapipe-stats-calculation tool
"""

from traitlets.config.loader import Config

from ctapipe.core import run_tool
from ctapipe.io import read_table
from ctapipe.tools.stats_calculation import StatisticsCalculatorTool


def test_stats_calc_tool(tmp_path, dl1_image_file):
    """check statistics calculation from DL1a files"""

    # Create a configuration suitable for the test
    tel_id = 3
    config = Config(
        {
            "StatisticsCalculatorTool": {
                "allowed_tels": [tel_id],
                "dl1a_column_name": "image",
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
