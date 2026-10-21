"""
Tests for CalibrationCalculator and related functions
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table, vstack
from astropy.time import Time
from traitlets.config.loader import Config

from ctapipe.monitoring.aggregator import HistogramAggregator
from ctapipe.monitoring.calculator import PixelStatisticsCalculator


def test_statistics_calculator(example_subarray):
    """test basic functionality of the PixelStatisticsCalculator"""

    # Create dummy data for testing
    n_images = 5050
    times = Time(
        np.linspace(60117.911, 60117.9258, num=n_images), scale="tai", format="mjd"
    )
    event_ids = np.linspace(35, 725000, num=n_images, dtype=int)
    rng = np.random.default_rng(0)
    charge_data = rng.normal(77.0, 10.0, size=(n_images, 2, 1855))
    # Create tables
    charge_table = Table(
        [times, event_ids, charge_data],
        names=("time", "event_id", "image"),
    )
    # Create configuration
    chunk_size = 1000
    chunk_shift = 500
    config = Config(
        {
            "PixelStatisticsCalculator": {
                "stats_aggregator_type": [
                    ("id", 1, "SigmaClippingAggregator"),
                ],
                "faulty_pixels_fraction": 0.09,
            },
            "SigmaClippingAggregator": {
                "chunking_type": "SizeChunking",
            },
            "SizeChunking": {
                "chunk_size": chunk_size,
                "chunk_shift": chunk_shift,
            },
        }
    )
    # Initialize the calculator from config
    calculator = PixelStatisticsCalculator(subarray=example_subarray, config=config)
    # Compute the statistical values
    stats = calculator.first_pass(table=charge_table, tel_id=1)
    # Set all chunks as faulty to aggregate the statistic values with a "global" chunk shift
    valid_chunks = np.zeros_like(stats["is_valid"].data, dtype=bool)
    # Run the second pass over the data
    stats_chunk_shift = calculator.second_pass(
        table=charge_table, valid_chunks=valid_chunks, tel_id=1
    )
    # Stack the statistic values from the first and second pass
    stats_stacked = vstack([stats, stats_chunk_shift])
    # Sort the stacked aggregated statistic values by starting time
    stats_stacked.sort(["time_start"])
    # Check if the calculated statistical values are reasonable
    # for a camera with two gain channels
    np.testing.assert_allclose(stats[0]["mean"], 77.0, atol=2.5)
    np.testing.assert_allclose(stats[1]["median"], 77.0, atol=2.5)
    np.testing.assert_allclose(stats[0]["std"], 10.0, atol=2.5)
    np.testing.assert_allclose(stats_chunk_shift[0]["mean"], 77.0, atol=2.5)
    np.testing.assert_allclose(stats_chunk_shift[1]["median"], 77.0, atol=2.5)
    np.testing.assert_allclose(stats_chunk_shift[0]["std"], 10.0, atol=2.5)
    # Check if overlapping chunks of the second pass were aggregated
    assert stats_chunk_shift is not None
    # Check if the number of aggregated chunks is correct
    # In the first pass, the number of chunks is equal to the
    # number of images divided by the chunk size plus one
    # overlapping chunk at the end.
    expected_len_firstpass = n_images // chunk_size + 1
    assert len(stats) == expected_len_firstpass
    # Only new windows are added: starts at rows 500, 1500, 2500 and 3500.
    # Aligned windows and the final overlap already exist in the first pass.
    expected_len_secondpass = 4
    assert len(stats_chunk_shift) == expected_len_secondpass
    np.testing.assert_array_equal(
        stats_chunk_shift["event_id_start"], event_ids[[500, 1500, 2500, 3500]]
    )
    # The total number of aggregated chunks is the sum of the
    # number of chunks in the first and second pass.
    assert len(stats_stacked) == expected_len_firstpass + expected_len_secondpass


def test_statistics_calculator_with_histogram_aggregator(example_subarray):
    """test that PixelStatisticsCalculator works with HistogramAggregator"""
    pytest.importorskip("hist")

    n_images = 120
    times = Time(
        np.linspace(60117.911, 60117.913, num=n_images), scale="tai", format="mjd"
    )
    event_ids = np.arange(n_images)
    rng = np.random.default_rng(7)
    charge_data = rng.normal(77.0, 10.0, size=(n_images, 2, 16))

    charge_table = Table(
        [times, event_ids, charge_data],
        names=("time", "event_id", "image"),
    )

    config = Config(
        {
            "PixelStatisticsCalculator": {
                "stats_aggregator_type": [
                    ("id", 1, "HistogramAggregator"),
                ],
                "outlier_detector_list": [
                    {
                        "apply_to": "histogram",
                        "name": "RangeOutlierDetector",
                        "config": {
                            "validity_range": [60.0, 90.0],
                        },
                    },
                ],
            },
            "HistogramAggregator": {
                "chunking_type": "SizeChunking",
                "axis_definition": {
                    "class_name": "Regular",
                    "bins": 30,
                    "start": 0.0,
                    "stop": 140.0,
                    "name": "image",
                },
                "axis_names": ["channel_id", "pixel_id"],
            },
            "SizeChunking": {
                "chunk_size": 60,
            },
        }
    )

    calculator = PixelStatisticsCalculator(subarray=example_subarray, config=config)
    stats = calculator.first_pass(table=charge_table, tel_id=1)

    assert len(stats) == 2
    assert stats[0]["histogram"].shape == (30, 2, 16)
    assert stats[0]["n_events"].shape == (2, 16)
    assert stats[0].meta["bin_edges"].shape == (31,)
    hist_object = HistogramAggregator.hist_from_tablerow(stats[0])
    assert [axis.name for axis in hist_object.axes] == [
        "image",
        "channel_id",
        "pixel_id",
    ]


def test_outlier_detector(example_subarray):
    """test the chunk shift option and the boundary case for the last chunk"""

    # Create dummy data for testing
    times = Time(
        np.linspace(60117.911, 60117.9258, num=5500), scale="tai", format="mjd"
    )
    event_ids = np.linspace(35, 725000, num=5500, dtype=int)
    rng = np.random.default_rng(0)
    ped_data = rng.normal(2.0, 5.0, size=(5500, 2, 1855))
    # Create table
    ped_table = Table(
        [times, event_ids, ped_data],
        names=("time", "event_id", "image"),
    )
    # Create configuration
    config = Config(
        {
            "PixelStatisticsCalculator": {
                "stats_aggregator_type": [
                    ("id", 1, "SigmaClippingAggregator"),
                ],
                "outlier_detector_list": [
                    {
                        "apply_to": "mean",
                        "name": "MedianOutlierDetector",
                        "config": {
                            "median_range_factors": [-3.0, 3.0],
                        },
                    },
                    {
                        "apply_to": "median",
                        "name": "StdOutlierDetector",
                        "config": {
                            "std_range_factors": [-2.0, 2.0],
                        },
                    },
                    {
                        "apply_to": "std",
                        "name": "RangeOutlierDetector",
                        "config": {
                            "validity_range": [2.0, 8.0],
                        },
                    },
                ],
                "faulty_pixels_fraction": 0.09,
            },
            "SigmaClippingAggregator": {
                "chunking_type": "SizeChunking",
            },
            "SizeChunking": {
                "chunk_size": 1000,
                "chunk_shift": 500,
            },
        }
    )
    # Initialize the calculator from config
    calculator = PixelStatisticsCalculator(subarray=example_subarray, config=config)
    # Run the first pass over the data
    stats_first_pass = calculator.first_pass(table=ped_table, tel_id=1)
    # Run the second pass over the data
    stats_second_pass = calculator.second_pass(
        table=ped_table, valid_chunks=stats_first_pass["is_valid"].data, tel_id=1
    )
    # Stack the statistic values from the first and second pass
    stats_stacked = vstack([stats_first_pass, stats_second_pass])
    # Sort the stacked aggregated statistic values by starting time
    stats_stacked.sort(["time_start"])
    # Check if overlapping chunks of the second pass were aggregated
    assert stats_second_pass is not None
    assert len(stats_stacked) > len(stats_second_pass)
    # Check if the calculated statistical values are reasonable
    # for a camera with two gain channels
    np.testing.assert_allclose(stats_stacked[0]["mean"], 2.0, atol=2.5)
    np.testing.assert_allclose(stats_stacked[1]["median"], 2.0, atol=2.5)
    np.testing.assert_allclose(stats_stacked[0]["std"], 5.0, atol=2.5)


@pytest.mark.parametrize("chunking_type", ["SizeChunking", "TimeChunking"])
@pytest.mark.parametrize("last_chunk_policy", ["overlap", "truncate", "skip"])
@pytest.mark.parametrize("faulty_index", [0, 2, -1])
def test_second_pass_chunk_boundaries(
    example_subarray, chunking_type, last_chunk_policy, faulty_index
):
    """Refinement follows actual chunks, including irregular event spacing."""
    # Use a varying event rate so time windows contain different numbers of rows.
    offsets = np.cumsum(np.tile([0.1, 0.15, 0.35, 0.4], 30))
    table = Table(
        {
            "time": Time("2020-01-01") + offsets * u.s,
            "event_id": np.arange(len(offsets)),
            "image": np.ones((len(offsets), 1, 2)) * u.V,
        }
    )
    settings = (
        {"chunk_size": 20, "chunk_shift": 10}
        if chunking_type == "SizeChunking"
        else {"chunk_duration": 5 * u.s, "chunk_shift": 2 * u.s}
    )
    calculator = PixelStatisticsCalculator(
        subarray=example_subarray,
        config=Config(
            {
                "PixelStatisticsCalculator": {
                    "stats_aggregator_type": "PlainAggregator"
                },
                "PlainAggregator": {"chunking_type": chunking_type},
                chunking_type: {**settings, "last_chunk_policy": last_chunk_policy},
            }
        ),
    )
    aggregator = calculator.stats_aggregators["PlainAggregator"]
    first = calculator.first_pass(table, tel_id=1)
    assert first["is_valid"].shape == (len(first),)
    assert aggregator.chunking.chunk_shift == settings["chunk_shift"]
    valid = np.ones(len(first), dtype=bool)
    valid[faulty_index] = False
    second = calculator.second_pass(table, valid, tel_id=1)
    assert len(second) > 0
    assert second["mean"].unit == u.V
    np.testing.assert_allclose(second["mean"], 1)

    faulty = first[~valid][0]
    assert np.all(second["time_start"] <= faulty["time_end"])
    assert np.all(second["time_end"] >= faulty["time_start"])
    for row in second:
        assert not np.any(
            (row["time_start"] == first["time_start"])
            & (row["time_end"] == first["time_end"])
        )
        events = table[
            (table["time"] >= row["time_start"]) & (table["time"] <= row["time_end"])
        ]
        assert np.all(row["n_events"] == len(events))


@pytest.mark.parametrize("chunking_type", ["SizeChunking", "TimeChunking"])
def test_second_pass_no_additional_chunks(example_subarray, chunking_type):
    """A single undersized faulty chunk has no new refinement windows."""
    table = Table(
        {
            "time": Time("2020-01-01") + np.arange(3) * u.s,
            "image": np.ones((3, 1, 2)),
        }
    )
    settings = (
        {"chunk_size": 10, "chunk_shift": 5}
        if chunking_type == "SizeChunking"
        else {"chunk_duration": 10 * u.s, "chunk_shift": 5 * u.s}
    )
    calculator = PixelStatisticsCalculator(
        subarray=example_subarray,
        config=Config(
            {
                "PixelStatisticsCalculator": {
                    "stats_aggregator_type": "PlainAggregator"
                },
                "PlainAggregator": {"chunking_type": chunking_type},
                chunking_type: {**settings, "allow_undersized_tables": True},
            }
        ),
    )
    first = calculator.first_pass(table, tel_id=1)
    second = calculator.second_pass(table, np.zeros(len(first), dtype=bool), tel_id=1)
    assert len(second) == 0
    assert second.colnames == first.colnames
