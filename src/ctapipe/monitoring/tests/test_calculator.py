"""
Tests for CalibrationCalculator and related functions
"""

import numpy as np
from astropy.table import Table, vstack
from astropy.time import Time
from traitlets.config.loader import Config

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
        names=("time_mono", "event_id", "image"),
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
                "chunk_shift": chunk_shift,
                "faulty_pixels_fraction": 0.09,
            },
            "SigmaClippingAggregator": {
                "chunk_size": chunk_size,
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
    # In the second pass, the number of chunks is equal to the
    # number of images divided by the chunk shift minus the
    # number of chunks in the first pass, since we set all
    # chunks to be faulty.
    expected_len_secondpass = (n_images // chunk_shift) - expected_len_firstpass
    assert len(stats_chunk_shift) == expected_len_secondpass
    # The total number of aggregated chunks is the sum of the
    # number of chunks in the first and second pass.
    assert len(stats_stacked) == expected_len_firstpass + expected_len_secondpass


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
        names=("time_mono", "event_id", "image"),
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
                "chunk_shift": 500,
                "faulty_pixels_fraction": 0.09,
            },
            "SigmaClippingAggregator": {
                "chunk_size": 1000,
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
