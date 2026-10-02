#!/usr/bin/env python3
"""
Test ctapipe-calculate-pixel-statistics tool
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table
from traitlets.config.loader import Config

from ctapipe.containers import ChunkHistogramContainer
from ctapipe.core import run_tool
from ctapipe.core.tool import ToolConfigurationError
from ctapipe.instrument import SubarrayDescription
from ctapipe.io import HDF5TableReader, TableLoader, read_table
from ctapipe.io.hdf5dataformat import (
    DL1_COLUMN_NAMES,
    DL1_PIXEL_HISTOGRAMS_GROUP,
    DL1_PIXEL_STATISTICS_GROUP,
)
from ctapipe.monitoring import HistogramAggregator
from ctapipe.monitoring.calculator import PixelStatisticsCalculator
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


def test_calculate_pixel_stats_tool_with_histogram_aggregator(tmp_path, dl1_image_file):
    """check tool execution with HistogramAggregator"""
    hist = pytest.importorskip("hist")

    tel_id = 3
    output_file = tmp_path / "subarray_image_hist_monitoring.dl1.h5"
    config = Config(
        {
            "PixelStatisticsCalculatorTool": {
                "allowed_tels": [3],
                "input_column_name": "image",
            },
            "PixelStatisticsCalculator": {
                "stats_aggregator_type": [
                    ("type", "*", "HistogramAggregator"),
                ],
                "outlier_detector_list": [
                    {
                        "apply_to": "histogram",
                        "name": "RangeOutlierDetector",
                        "config": {
                            "validity_range": [0.0, 100.0],
                        },
                    },
                ],
            },
            "HistogramAggregator": {
                "chunking_type": "SizeChunking",
                "axis_definition": {
                    "class_name": "Regular",
                    "bins": 20,
                    "start": 0.0,
                    "stop": 200.0,
                },
            },
            "SizeChunking": {
                "chunk_size": 1,
            },
        }
    )

    run_tool(
        PixelStatisticsCalculatorTool(config=config),
        argv=[
            f"--input_url={dl1_image_file}",
            f"--output_path={output_file}",
            "--overwrite",
        ],
        cwd=tmp_path,
        raises=True,
    )

    stats = read_table(
        output_file,
        path=f"{DL1_PIXEL_HISTOGRAMS_GROUP}/subarray_image/tel_{tel_id:03d}",
    )

    assert "histogram" in stats.colnames
    assert "bin_edges" in stats.meta
    assert stats["histogram"].ndim == 4

    with HDF5TableReader(output_file) as reader:
        for i, container in enumerate(
            reader.read(
                table_name=f"{DL1_PIXEL_HISTOGRAMS_GROUP}/subarray_image/tel_{tel_id:03d}",
                containers=ChunkHistogramContainer,
                prefixes=[""],
            )
        ):
            h = HistogramAggregator.hist_from_container(container)
            assert isinstance(h, hist.Hist)


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


def test_calculate_pixel_stats_tool_per_gain(tmp_path, dl1_image_file):
    """check per-gain statistics calculation from gain selected image data"""

    tel_id = 3
    output_file = tmp_path / "per_gain_monitoring.dl1.h5"
    config = Config(
        {
            "PixelStatisticsCalculatorTool": {
                "allowed_tels": [tel_id],
            },
            "PixelStatisticsCalculator": {
                "stats_aggregator_type": [
                    ("type", "*", "PlainAggregator"),
                ],
            },
            "SizeChunking": {
                "chunk_size": 1,
            },
        }
    )
    run_tool(
        PixelStatisticsCalculatorTool(config=config),
        argv=[
            f"--input_url={dl1_image_file}",
            f"--output_path={output_file}",
            "--per-gain",
            "--overwrite",
        ],
        cwd=tmp_path,
        raises=True,
    )

    with TableLoader(dl1_image_file) as loader:
        dl1_table = loader.read_telescope_events(telescopes=[tel_id], dl1_images=True)
    stats = read_table(
        output_file,
        path=f"{DL1_PIXEL_STATISTICS_GROUP}/subarray_image/tel_{tel_id:03d}",
    )

    n_pixels = dl1_table["image"].shape[1]
    assert stats["mean"].shape == (len(dl1_table), 2, n_pixels)
    # Each sample is filled into exactly one gain channel
    np.testing.assert_array_equal(stats["n_events"].sum(axis=1), 1)

    gain = dl1_table["selected_gain_channel"]
    event_index, pixel_index = np.indices(gain.shape)
    np.testing.assert_allclose(
        stats["mean"][event_index, gain, pixel_index], dl1_table["image"]
    )
    assert np.all(np.isnan(stats["mean"][event_index, 1 - gain, pixel_index]))


@pytest.mark.parametrize("per_gain_statistics", [True, False])
def test_reshape_dl1_dimensions_per_gain(dl1_image_file, per_gain_statistics):
    """check the reshaping of gain selected data with mixed gain channels"""

    tel_id = 3
    subarray = SubarrayDescription.from_hdf(dl1_image_file)
    n_pixels = subarray.tel[tel_id].camera.geometry.n_pixels
    n_events = 5

    rng = np.random.default_rng(0)
    gain = rng.integers(0, 2, size=(n_events, n_pixels), dtype=np.int8)
    image = rng.normal(10.0, 1.0, size=(n_events, n_pixels)).astype(np.float32)
    dl1_table = Table(
        {
            "image": image * u.ct,
            "peak_time": image * u.ns,
            "selected_gain_channel": gain,
        }
    )

    tool = PixelStatisticsCalculatorTool(per_gain_statistics=per_gain_statistics)
    tool.subarray = subarray
    tool._reshape_dl1_dimensions(dl1_table, tel_id)

    if not per_gain_statistics:
        assert dl1_table["image"].shape == (n_events, 1, n_pixels)
        return

    event_index, pixel_index = np.indices(gain.shape)
    for col, unit in [("image", u.ct), ("peak_time", u.ns)]:
        assert dl1_table[col].shape == (n_events, 2, n_pixels)
        assert dl1_table[col].unit == unit
        np.testing.assert_array_equal(
            dl1_table[col][event_index, gain, pixel_index], image
        )
        assert np.all(np.isnan(dl1_table[col][event_index, 1 - gain, pixel_index]))


def test_per_gain_missing_selected_gain_channel(dl1_image_file):
    """check error if per-gain statistics are requested without gain information"""

    tel_id = 3
    subarray = SubarrayDescription.from_hdf(dl1_image_file)
    n_pixels = subarray.tel[tel_id].camera.geometry.n_pixels
    dl1_table = Table({"image": np.zeros((5, n_pixels), dtype=np.float32)})

    tool = PixelStatisticsCalculatorTool(
        config=Config({"SizeChunking": {"chunk_size": 1}}),
        per_gain_statistics=True,
    )
    tool.subarray = subarray
    tool.stats_calculator = PixelStatisticsCalculator(parent=tool, subarray=subarray)

    with pytest.raises(
        ToolConfigurationError, match="'selected_gain_channel' not found"
    ):
        tool._is_valid_table(dl1_table, tel_id)
