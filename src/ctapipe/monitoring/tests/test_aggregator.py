"""
Tests for StatisticsAggregator and related functions
"""

import numpy as np
import pytest
from astropy.table import Table
from astropy.time import Time

from ctapipe.monitoring.aggregator import PlainAggregator, SigmaClippingAggregator


def test_aggregators(example_subarray):
    """test basic functionality of the StatisticsAggregators"""

    # Create dummy data for testing
    times = Time(
        np.linspace(60117.911, 60117.9258, num=5000), scale="tai", format="mjd"
    )
    event_ids = np.linspace(35, 725000, num=5000, dtype=int)
    rng = np.random.default_rng(0)
    ped_data = rng.normal(2.0, 5.0, size=(5000, 2, 1855))
    charge_data = rng.normal(77.0, 10.0, size=(5000, 2, 1855))
    time_data = rng.normal(18.0, 5.0, size=(5000, 2, 1855))
    # Create tables
    ped_table = Table(
        [times, event_ids, ped_data],
        names=("time_mono", "event_id", "image"),
    )
    charge_table = Table(
        [times, event_ids, charge_data],
        names=("time_mono", "event_id", "image"),
    )
    time_table = Table(
        [times, event_ids, time_data],
        names=("time_mono", "event_id", "peak_time"),
    )
    # Initialize the aggregators
    chunk_size = 2500
    ped_aggregator = SigmaClippingAggregator(
        subarray=example_subarray, chunk_size=chunk_size
    )
    ff_charge_aggregator = SigmaClippingAggregator(
        subarray=example_subarray, chunk_size=chunk_size
    )
    ff_time_aggregator = PlainAggregator(
        subarray=example_subarray, chunk_size=chunk_size
    )

    # Compute the statistical values
    ped_stats = ped_aggregator(table=ped_table)
    charge_stats = ff_charge_aggregator(table=charge_table)
    time_stats = ff_time_aggregator(table=time_table, col_name="peak_time")

    # Check if the start and end values are properly set for the timestamps and event IDs
    assert ped_stats[0]["time_start"] == times[0]
    assert time_stats[0]["event_id_start"] == event_ids[0]
    assert ped_stats[1]["time_end"] == times[-1]
    assert time_stats[1]["event_id_end"] == event_ids[-1]

    # Check n_events behavior:
    # - For PlainAggregator: should equal chunk_size for all pixels
    # - For SigmaClippingAggregator: should be <= chunk_size (some events clipped)
    # - Should have the right shape (2 gain channels, 1855 pixels)

    # PlainAggregator should have all events (no clipping)
    assert np.all(time_stats[0]["n_events"] == chunk_size)
    assert time_stats[0]["n_events"].shape == (2, 1855)

    # SigmaClippingAggregator should have fewer or equal events (due to clipping)
    assert np.all(ped_stats[0]["n_events"] <= chunk_size)
    assert np.all(charge_stats[0]["n_events"] <= chunk_size)
    # Most pixels should still have close to the full chunk size (normal data)
    assert np.mean(ped_stats[0]["n_events"]) > 0.9 * chunk_size
    assert np.mean(charge_stats[0]["n_events"]) > 0.9 * chunk_size
    # Check shape
    assert ped_stats[0]["n_events"].shape == (2, 1855)
    assert charge_stats[0]["n_events"].shape == (2, 1855)

    # Check if the calculated statistical values are reasonable
    # for a camera with two gain channels
    np.testing.assert_allclose(ped_stats[0]["mean"], 2.0, atol=1.5)
    np.testing.assert_allclose(charge_stats[0]["mean"], 77.0, atol=1.5)
    np.testing.assert_allclose(time_stats[0]["mean"], 18.0, atol=1.5)

    np.testing.assert_allclose(ped_stats[1]["median"], 2.0, atol=1.5)
    np.testing.assert_allclose(charge_stats[1]["median"], 77.0, atol=1.5)
    np.testing.assert_allclose(time_stats[1]["median"], 18.0, atol=1.5)

    np.testing.assert_allclose(ped_stats[0]["std"], 5.0, atol=1.5)
    np.testing.assert_allclose(charge_stats[0]["std"], 10.0, atol=1.5)
    np.testing.assert_allclose(time_stats[0]["std"], 5.0, atol=1.5)


def test_chunk_shift(example_subarray):
    """test the chunk shift option and the boundary case for the last chunk"""

    # Create dummy data for testing
    times = Time(
        np.linspace(60117.911, 60117.9258, num=5500), scale="tai", format="mjd"
    )
    event_ids = np.linspace(35, 725000, num=5500, dtype=int)
    rng = np.random.default_rng(0)
    charge_data = rng.normal(77.0, 10.0, size=(5500, 2, 1855))
    # Create table
    charge_table = Table(
        [times, event_ids, charge_data],
        names=("time_mono", "event_id", "image"),
    )
    # Initialize the aggregator
    aggregator = SigmaClippingAggregator(subarray=example_subarray, chunk_size=2500)
    # Compute aggregated statistic values
    chunk_stats = aggregator(table=charge_table)
    chunk_stats_shift = aggregator(table=charge_table, chunk_shift=2000)
    # Check if three chunks are used for the computation of aggregated statistic values as the last chunk overflows
    assert len(chunk_stats) == 3
    # Check if two chunks are used for the computation of aggregated statistic values as the last chunk is dropped
    assert len(chunk_stats_shift) == 2
    # Check if ValueError is raised when the chunk_size is larger than the length of table
    with pytest.raises(ValueError):
        _ = aggregator(table=charge_table[1000:1500])
    # Check if ValueError is raised when the chunk_shift is smaller than the chunk_size
    with pytest.raises(ValueError):
        _ = aggregator(table=charge_table, chunk_shift=3000)


def test_with_outliers(example_subarray):
    """test the robustness of the aggregators in the presence of outliers"""

    # Create dummy data for testing
    times = Time(
        np.linspace(60117.911, 60117.9258, num=5000), scale="tai", format="mjd"
    )
    event_ids = np.linspace(35, 725000, num=5000, dtype=int)
    rng = np.random.default_rng(0)
    ped_data = rng.normal(2.0, 5.0, size=(5000, 2, 1855))
    # Insert fake outliers that will skrew the mean value
    ped_data[12, 0, :] = 100000.0
    ped_data[16, 0, :] = 100000.0
    ped_data[18, 1, :] = 100000.0
    ped_data[28, 1, :] = 100000.0
    # Create table
    ped_table = Table(
        [times, event_ids, ped_data],
        names=("time_mono", "event_id", "image"),
    )
    # Initialize the aggregators
    sigmaclipping_aggregator = SigmaClippingAggregator(
        subarray=example_subarray, chunk_size=2500
    )
    plain_aggregator = PlainAggregator(subarray=example_subarray, chunk_size=2500)

    # Compute aggregated statistic values
    sigmaclipping_chunk_stats = sigmaclipping_aggregator(table=ped_table)
    plain_chunk_stats = plain_aggregator(table=ped_table)

    # Check if SigmaClippingAggregator is robust to a few fake outliers as expected
    np.testing.assert_allclose(sigmaclipping_chunk_stats[0]["mean"], 2.0, atol=1.5)

    # Check if PlainAggregator is not robust to a few fake outliers as expected
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(plain_chunk_stats[0]["mean"], 2.0, atol=1.5)

    # Check that sigma clipping actually removed some events due to outliers
    # PlainAggregator should have all 2500 events per pixel
    assert np.all(plain_chunk_stats[0]["n_events"] == 2500)

    # SigmaClippingAggregator should have removed outliers, so fewer events
    # At least the pixels with outliers should have fewer than 2500 events
    sigma_n_events = sigmaclipping_chunk_stats[0]["n_events"]
    assert np.any(sigma_n_events < 2500)  # Some pixels should have clipped events

    # Specifically, gain channel 0 should have fewer events due to outliers at events 12, 16
    # and gain channel 1 should have fewer events due to outliers at events 18, 28
    assert np.all(sigma_n_events[0, :] <= 2500)  # All pixels in gain 0 <= 2500
    assert np.all(sigma_n_events[1, :] <= 2500)  # All pixels in gain 1 <= 2500
    # Most pixels without outliers should still have close to 2500 events
    assert np.mean(sigma_n_events) > 0.99 * 2500
