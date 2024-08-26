"""
Tests for CalibrationCalculator and related functions
"""

import numpy as np
from astropy.table import Table
from astropy.time import Time
from traitlets.config.loader import Config

from ctapipe.monitoring.aggregator import PlainAggregator
from ctapipe.monitoring.calculator import CalibrationCalculator, StatisticsCalculator


def test_onepass_calculator(example_subarray):
    """test basic 'one pass' functionality of the StatisticsCalculator"""

    # Create dummy data for testing
    times = Time(
        np.linspace(60117.911, 60117.9258, num=5000), scale="tai", format="mjd"
    )
    event_ids = np.linspace(35, 725000, num=5000, dtype=int)
    rng = np.random.default_rng(0)
    charge_data = rng.normal(77.0, 10.0, size=(5000, 2, 1855))
    # Create tables
    charge_table = Table(
        [times, event_ids, charge_data],
        names=("time_mono", "event_id", "image"),
    )
    # Initialize the aggregators and calculators
    chunk_size = 500
    aggregator = PlainAggregator(subarray=example_subarray, chunk_size=chunk_size)
    calculator = CalibrationCalculator.from_name(
        name="StatisticsCalculator",
        subarray=example_subarray,
        stats_aggregator=aggregator,
    )
    calculator_chunk_shift = StatisticsCalculator(
        subarray=example_subarray, stats_aggregator=aggregator, chunk_shift=250
    )
    # Compute the statistical values
    stats = calculator(table=charge_table, tel_id=1)
    stats_chunk_shift = calculator_chunk_shift(table=charge_table, tel_id=1)

    # Check if the calculated statistical values are reasonable
    # for a camera with two gain channels
    np.testing.assert_allclose(stats[0]["mean"], 77.0, atol=2.5)
    np.testing.assert_allclose(stats[1]["median"], 77.0, atol=2.5)
    np.testing.assert_allclose(stats[0]["std"], 10.0, atol=2.5)
    # Check if three chunks are used for the computation of aggregated statistic values as the last chunk overflows
    assert len(stats) * 2 == len(stats_chunk_shift) + 1

def test_secondpass_calculator(example_subarray):
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
            "StatisticsCalculator": {
                "stats_aggregator_type": [
                    ("id", 1, "SigmaClippingAggregator"),
                ],
                "outlier_detector_list": [
                    {
                        "apply_to": "mean",
                        "name": "StdOutlierDetector",
                        "validity_range": [-2.0, 2.0],
                    },
                    {
                        "apply_to": "median",
                        "name": "StdOutlierDetector",
                        "validity_range": [-3.0, 3.0],
                    },
                    {
                        "apply_to": "std",
                        "name": "RangeOutlierDetector",
                        "validity_range": [2.0, 8.0],
                    },
                ],
                "chunk_shift": 100,
                "second_pass": True,
                "faulty_pixels_threshold": 1.0,
            },
            "SigmaClippingAggregator": {
                "chunk_size": 500,
            },
        }
    )
    # Initialize the calculator from config
    calculator = StatisticsCalculator(subarray=example_subarray, config=config)
    # Compute aggregated statistic values
    stats = calculator(ped_table, 1, col_name="image")
    # Check if the second pass was activated
    assert len(stats) > 20
   
