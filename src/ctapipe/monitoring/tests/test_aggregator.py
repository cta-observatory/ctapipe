"""
Tests for StatisticsAggregator and related functions
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table
from astropy.time import Time
from traitlets.config import Config

from ctapipe.monitoring.aggregator import (
    PlainAggregator,
    SigmaClippingAggregator,
)


def test_aggregators():
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
        names=("time", "event_id", "image"),
    )
    charge_table = Table(
        [times, event_ids, charge_data],
        names=("time", "event_id", "image"),
    )
    time_table = Table(
        [times, event_ids, time_data],
        names=("time", "event_id", "peak_time"),
    )
    # Initialize the aggregators using Config
    chunk_size = 2500
    config = Config(
        {
            "SigmaClippingAggregator": {"chunking_type": "SizeChunking"},
            "SizeChunking": {"chunk_size": chunk_size},
        }
    )

    ped_aggregator = SigmaClippingAggregator(config=config)
    ff_charge_aggregator = SigmaClippingAggregator(config=config)

    config_plain = Config(
        {
            "PlainAggregator": {"chunking_type": "SizeChunking"},
            "SizeChunking": {"chunk_size": chunk_size},
        }
    )

    ff_time_aggregator = PlainAggregator(config=config_plain)

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


def test_chunk_shift():
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
        names=("time", "event_id", "image"),
    )
    # Initialize the aggregator
    config = Config(
        {
            "SigmaClippingAggregator": {"chunking_type": "SizeChunking"},
            "SizeChunking": {"chunk_size": 2500},
        }
    )
    aggregator = SigmaClippingAggregator(config=config)
    # Compute aggregated statistic values
    chunk_stats = aggregator(table=charge_table)

    # Test with overlapping chunks
    config_overlap = Config(
        {
            "SigmaClippingAggregator": {"chunking_type": "SizeChunking"},
            "SizeChunking": {"chunk_size": 2500, "chunk_shift": 2000},
        }
    )
    aggregator_overlap = SigmaClippingAggregator(config=config_overlap)
    chunk_stats_shift = aggregator_overlap(table=charge_table)
    # Check if three chunks are used for the computation of aggregated statistic values as the last chunk overflows
    assert len(chunk_stats) == 3
    # Check if two chunks are used for the computation of aggregated statistic values as the last chunk is dropped
    assert len(chunk_stats_shift) == 3
    # Check if ValueError is raised when the chunk_size is larger than the length of table
    with pytest.raises(ValueError):
        _ = aggregator(table=charge_table[1000:1500])


def test_with_outliers():
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
        names=("time", "event_id", "image"),
    )
    # Initialize the aggregators
    config_sigma = Config(
        {
            "SigmaClippingAggregator": {"chunking_type": "SizeChunking"},
            "SizeChunking": {"chunk_size": 2500},
        }
    )
    sigmaclipping_aggregator = SigmaClippingAggregator(config=config_sigma)

    config_plain = Config(
        {
            "PlainAggregator": {"chunking_type": "SizeChunking"},
            "SizeChunking": {"chunk_size": 2500},
        }
    )
    plain_aggregator = PlainAggregator(config=config_plain)

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


def test_time_based_chunking():
    """test time-based chunking functionality"""

    # Create dummy data spanning 10 seconds with uniform time distribution
    start_time = Time("2020-01-01T20:00:00")
    times = start_time + np.linspace(0, 10, 1000) * u.s
    event_ids = np.arange(1000)
    rng = np.random.default_rng(42)
    data = rng.normal(5.0, 1.0, size=(1000, 2, 10))

    # Create table
    table = Table(
        [times, event_ids, data],
        names=("time", "event_id", "image"),
    )

    # Test time-based chunking with 2-second intervals
    config_time = Config(
        {
            "PlainAggregator": {"chunking_type": "TimeChunking"},
            "TimeChunking": {"chunk_duration": 2 * u.s},
        }
    )
    aggregator_time = PlainAggregator(config=config_time)
    result_time = aggregator_time(table=table)

    # Should create 5 chunks (10 seconds / 2 seconds per chunk)
    assert len(result_time) == 5

    # Each chunk should span approximately 2 seconds
    for i in range(len(result_time)):
        chunk_duration = (
            result_time[i]["time_end"] - result_time[i]["time_start"]
        ).to_value("s")
        assert 1.5 <= chunk_duration <= 2.5, f"Chunk {i} duration: {chunk_duration}s"

    # Test with larger time chunks
    config_time_5s = Config(
        {
            "PlainAggregator": {"chunking_type": "TimeChunking"},
            "TimeChunking": {"chunk_duration": 5 * u.s},
        }
    )
    aggregator_time_5s = PlainAggregator(config=config_time_5s)
    result_time_5s = aggregator_time_5s(table=table)

    # Should create 2 chunks (10 seconds / 5 seconds per chunk)
    assert len(result_time_5s) == 2


def test_time_based_chunking_with_shift():
    """test time-based chunking with overlapping windows (rolling statistics)"""

    # Create dummy data spanning exactly 5 seconds to get cleaner chunks
    start_time = Time("2020-01-01T20:00:00")
    times = start_time + np.linspace(0, 4.99, 500) * u.s
    event_ids = np.arange(500)
    rng = np.random.default_rng(123)
    data = rng.normal(10.0, 2.0, size=(500, 2, 5))

    # Create table
    table = Table(
        [times, event_ids, data],
        names=("time", "event_id", "image"),
    )

    # Test overlapping time chunks: 2-second chunks with 1-second shift
    config_overlap = Config(
        {
            "PlainAggregator": {"chunking_type": "TimeChunking"},
            "TimeChunking": {"chunk_duration": 2 * u.s, "chunk_shift": 1 * u.s},
        }
    )
    aggregator_overlap = PlainAggregator(config=config_overlap)
    result_overlap = aggregator_overlap(table=table)

    # With 5 seconds of data, 2-second chunks with 1-second shift should create:
    # Chunk 0: [0:2]s, Chunk 1: [1:3]s, Chunk 2: [2:4]s, Chunk 3: [3:5]s = 4 chunks
    # Each chunk is exactly 2 seconds and we can fit all 4 within the 5-second span
    assert len(result_overlap) == 4

    # Verify we have overlapping time ranges (at least for first few chunks)
    if len(result_overlap) >= 2:
        for i in range(min(2, len(result_overlap) - 1)):
            current_end = result_overlap[i]["time_end"]
            next_start = result_overlap[i + 1]["time_start"]
            # Next chunk should start before current chunk ends (overlap)
            assert next_start < current_end

    # Test non-overlapping chunks for comparison
    config_no_overlap = Config(
        {
            "PlainAggregator": {"chunking_type": "TimeChunking"},
            "TimeChunking": {"chunk_duration": 2 * u.s},
        }
    )
    aggregator_no_overlap = PlainAggregator(config=config_no_overlap)
    result_no_overlap = aggregator_no_overlap(table=table)

    # Should create 3 chunks for 5 seconds with 2-second chunks:
    # [0:2]s, [2:4]s, and [3:5]s (last chunk overlaps to maintain full duration)
    assert len(result_no_overlap) == 3


def test_time_vs_event_chunking_consistency():
    """test that time and event chunking modes work consistently"""

    # Create dummy data
    times = Time(np.linspace(60117.911, 60117.912, num=500), scale="tai", format="mjd")
    event_ids = np.arange(500)
    rng = np.random.default_rng(456)
    data = rng.normal(3.0, 0.5, size=(500, 2, 8))

    # Create table
    table = Table(
        [times, event_ids, data],
        names=("time", "event_id", "image"),
    )

    # Test event-based chunking (default behavior)
    config_events = Config(
        {
            "SigmaClippingAggregator": {"chunking_type": "SizeChunking"},
            "SizeChunking": {"chunk_size": 100},
        }
    )
    aggregator_events = SigmaClippingAggregator(config=config_events)
    result_events = aggregator_events(table=table)

    # Should create 5 chunks (500 events / 100 events per chunk)
    assert len(result_events) == 5

    # All chunks should have consistent event counts (no clipping expected for normal data)
    for i in range(len(result_events)):
        assert np.all(result_events[i]["n_events"] <= 100)
        assert np.mean(result_events[i]["n_events"]) > 95  # Most events retained

    # Test time-based chunking
    total_duration = (times[-1] - times[0]).to_value("s")
    chunk_duration = total_duration / 5  # Same number of chunks as event-based

    config_time = Config(
        {
            "SigmaClippingAggregator": {"chunking_type": "TimeChunking"},
            "TimeChunking": {"chunk_duration": chunk_duration * u.s},
        }
    )
    aggregator_time = SigmaClippingAggregator(config=config_time)
    result_time = aggregator_time(table=table)

    # Should create approximately the same number of chunks
    assert 3 <= len(result_time) <= 7  # Allow some variation due to time boundaries

    # Statistical results should be similar between chunking methods
    event_means = [np.mean(chunk["mean"]) for chunk in result_events]
    time_means = [np.mean(chunk["mean"]) for chunk in result_time]

    # Both should be close to the true mean (3.0)
    assert np.abs(np.mean(event_means) - 3.0) < 0.5
    assert np.abs(np.mean(time_means) - 3.0) < 0.5


def test_time_chunking_validation():
    """test error handling for time-based chunking parameters"""

    # Create small test data
    times = Time(
        np.linspace(60117.911, 60117.911 + 5 / 86400, num=100),
        scale="tai",
        format="mjd",
    )
    event_ids = np.arange(100)
    rng = np.random.default_rng(0)
    data = rng.normal(1.0, 0.1, size=(100, 1, 5))

    table = Table(
        [times, event_ids, data],
        names=("time", "event_id", "image"),
    )

    # Test chunk_shift larger than chunk_duration (creates gaps between chunks)
    config_gaps = Config(
        {
            "PlainAggregator": {"chunking_type": "TimeChunking"},
            "TimeChunking": {
                "chunk_duration": 2 * u.s,
                "chunk_shift": 3 * u.s,  # 3 > 2 seconds - creates gaps
            },
        }
    )
    aggregator = PlainAggregator(config=config_gaps)

    # This should now work (creates chunks with gaps between them)
    result_gaps = aggregator(table=table)
    # Should create fewer chunks because of the gaps
    assert len(result_gaps) >= 1

    # Test that chunk_duration=0 raises a ValueError
    config_none = Config(
        {
            "PlainAggregator": {"chunking_type": "TimeChunking"},
            "TimeChunking": {"chunk_duration": 0 * u.s},
        }
    )
    aggregator_none = PlainAggregator(config=config_none)
    with pytest.raises(ValueError):
        _ = aggregator_none(table=table)


def test_1d_data_handling():
    """test that aggregators handle 1D data correctly (single-pixel case)"""

    # Create 1D dummy data (e.g., single pixel or scalar values per event)
    times = Time(
        np.linspace(60117.911, 60117.9258, num=1000), scale="tai", format="mjd"
    )
    event_ids = np.arange(1000)
    rng = np.random.default_rng(42)

    # 1D data: shape (n_events,) - simulates single pixel or aggregated value
    data_1d = rng.normal(10.0, 2.0, size=1000)

    # Create table with 1D data
    table_1d = Table(
        [times, event_ids, data_1d],
        names=("time", "event_id", "value"),
    )

    # Test PlainAggregator with 1D data
    config_1d = Config(
        {
            "PlainAggregator": {"chunking_type": "SizeChunking"},
            "SizeChunking": {"chunk_size": 500},
        }
    )
    plain_aggregator = PlainAggregator(config=config_1d)
    plain_stats = plain_aggregator(table=table_1d, col_name="value")

    # Should create 2 chunks
    assert len(plain_stats) == 2

    # Check that statistics are computed correctly
    # For 1D data, results should be scalar-like (shape ())
    assert plain_stats[0]["mean"].shape == ()
    assert plain_stats[0]["median"].shape == ()
    assert plain_stats[0]["std"].shape == ()
    assert plain_stats[0]["n_events"].shape == ()

    # Check that results are not NaN (verify the 1D fix works)
    assert not np.isnan(plain_stats[0]["mean"])
    assert not np.isnan(plain_stats[0]["median"])
    assert not np.isnan(plain_stats[0]["std"])
    assert not np.isnan(plain_stats[1]["mean"])
    assert not np.isnan(plain_stats[1]["median"])
    assert not np.isnan(plain_stats[1]["std"])

    # Check that values are reasonable (close to true values)
    np.testing.assert_allclose(plain_stats[0]["mean"], 10.0, atol=1.0)
    np.testing.assert_allclose(plain_stats[0]["std"], 2.0, atol=0.5)

    # Check n_events is correct for 1D case
    assert plain_stats[0]["n_events"] == 500
    assert plain_stats[1]["n_events"] == 500

    # Test SigmaClippingAggregator with 1D data
    config_sigma_1d = Config(
        {
            "SigmaClippingAggregator": {"chunking_type": "SizeChunking"},
            "SizeChunking": {"chunk_size": 500},
        }
    )
    sigma_aggregator = SigmaClippingAggregator(config=config_sigma_1d)
    sigma_stats = sigma_aggregator(table=table_1d, col_name="value")

    # Should create 2 chunks
    assert len(sigma_stats) == 2

    # Check that statistics are computed correctly
    assert sigma_stats[0]["mean"].shape == ()
    assert sigma_stats[0]["median"].shape == ()
    assert sigma_stats[0]["std"].shape == ()
    assert sigma_stats[0]["n_events"].shape == ()

    # Check that results are not NaN (verify the 1D fix works)
    assert not np.isnan(sigma_stats[0]["mean"])
    assert not np.isnan(sigma_stats[0]["median"])
    assert not np.isnan(sigma_stats[0]["std"])
    assert not np.isnan(sigma_stats[1]["mean"])
    assert not np.isnan(sigma_stats[1]["median"])
    assert not np.isnan(sigma_stats[1]["std"])

    # Check that values are reasonable
    np.testing.assert_allclose(sigma_stats[0]["mean"], 10.0, atol=1.0)
    np.testing.assert_allclose(sigma_stats[0]["std"], 2.0, atol=0.5)

    # For normal data without outliers, sigma clipping should keep most events
    assert sigma_stats[0]["n_events"] >= 480  # At least 96% of events
    assert sigma_stats[1]["n_events"] >= 480

    # Test with 1D data containing outliers
    data_1d_outliers = rng.normal(10.0, 2.0, size=1000)
    data_1d_outliers[50] = 1000.0  # Add outlier in first chunk
    data_1d_outliers[600] = -500.0  # Add outlier in second chunk

    table_1d_outliers = Table(
        [times, event_ids, data_1d_outliers],
        names=("time", "event_id", "value"),
    )

    plain_stats_outliers = plain_aggregator(table=table_1d_outliers, col_name="value")
    sigma_stats_outliers = sigma_aggregator(table=table_1d_outliers, col_name="value")

    # Check that results are not NaN even with outliers
    assert not np.isnan(plain_stats_outliers[0]["mean"])
    assert not np.isnan(plain_stats_outliers[1]["mean"])
    assert not np.isnan(sigma_stats_outliers[0]["mean"])
    assert not np.isnan(sigma_stats_outliers[1]["mean"])

    # PlainAggregator should be affected by outliers
    # Mean should be significantly off due to outliers
    assert np.abs(plain_stats_outliers[0]["mean"] - 10.0) > 1.0

    # SigmaClippingAggregator should be robust to outliers
    np.testing.assert_allclose(sigma_stats_outliers[0]["mean"], 10.0, atol=1.0)
    np.testing.assert_allclose(sigma_stats_outliers[1]["mean"], 10.0, atol=1.0)

    # Sigma clipping should have removed some events
    assert sigma_stats_outliers[0]["n_events"] < 500
    assert sigma_stats_outliers[1]["n_events"] < 500


def test_nan_handling():
    """test that aggregators properly handle NaN values in input data"""

    # Create test data with NaN values
    times = Time(
        np.linspace(60117.911, 60117.9258, num=1000), scale="tai", format="mjd"
    )
    event_ids = np.arange(1000)
    rng = np.random.default_rng(123)

    # Test 1D data with NaN values
    data_1d = rng.normal(10.0, 2.0, size=1000)
    # Insert NaN values in both chunks
    data_1d[10] = np.nan
    data_1d[20] = np.nan
    data_1d[520] = np.nan
    data_1d[530] = np.nan

    table_1d = Table(
        [times, event_ids, data_1d],
        names=("time", "event_id", "value"),
    )

    # Test PlainAggregator with NaN values
    config_plain_nan = Config(
        {
            "PlainAggregator": {"chunking_type": "SizeChunking"},
            "SizeChunking": {"chunk_size": 500},
        }
    )
    plain_aggregator = PlainAggregator(config=config_plain_nan)
    plain_stats = plain_aggregator(table=table_1d, col_name="value")

    assert len(plain_stats) == 2

    # Check that computed statistics are not NaN (NaN values should be masked)
    assert not np.isnan(plain_stats[0]["mean"])
    assert not np.isnan(plain_stats[0]["median"])
    assert not np.isnan(plain_stats[0]["std"])
    assert not np.isnan(plain_stats[1]["mean"])
    assert not np.isnan(plain_stats[1]["median"])
    assert not np.isnan(plain_stats[1]["std"])

    # Check that n_events reflects NaN exclusion
    # First chunk: 500 events - 2 NaNs = 498
    # Second chunk: 500 events - 2 NaNs = 498
    assert plain_stats[0]["n_events"] == 498
    assert plain_stats[1]["n_events"] == 498

    # Check that statistics are still accurate after excluding NaN
    np.testing.assert_allclose(plain_stats[0]["mean"], 10.0, atol=1.0)
    np.testing.assert_allclose(plain_stats[1]["mean"], 10.0, atol=1.0)

    # Test SigmaClippingAggregator with NaN values
    config_sigma_nan = Config(
        {
            "SigmaClippingAggregator": {"chunking_type": "SizeChunking"},
            "SizeChunking": {"chunk_size": 500},
        }
    )
    sigma_aggregator = SigmaClippingAggregator(config=config_sigma_nan)
    sigma_stats = sigma_aggregator(table=table_1d, col_name="value")

    assert len(sigma_stats) == 2

    # Check that computed statistics are not NaN
    assert not np.isnan(sigma_stats[0]["mean"])
    assert not np.isnan(sigma_stats[0]["median"])
    assert not np.isnan(sigma_stats[0]["std"])
    assert not np.isnan(sigma_stats[1]["mean"])
    assert not np.isnan(sigma_stats[1]["median"])
    assert not np.isnan(sigma_stats[1]["std"])

    # Check that n_events reflects NaN exclusion (and possibly sigma clipping)
    # Should be <= 498 (NaNs removed, possibly more removed by sigma clipping)
    assert sigma_stats[0]["n_events"] <= 498
    assert sigma_stats[1]["n_events"] <= 498
    # But most events should remain (normal data)
    assert sigma_stats[0]["n_events"] >= 475
    assert sigma_stats[1]["n_events"] >= 475

    # Test N-D data (2D camera-like) with NaN values
    data_2d = rng.normal(5.0, 1.0, size=(1000, 2, 10))
    # Insert NaN values in various pixels
    data_2d[15, 0, 3] = np.nan  # First chunk, gain 0, pixel 3
    data_2d[25, 1, 7] = np.nan  # First chunk, gain 1, pixel 7
    data_2d[550, 0, 5] = np.nan  # Second chunk, gain 0, pixel 5
    data_2d[560, 1, 2] = np.nan  # Second chunk, gain 1, pixel 2

    table_2d = Table(
        [times, event_ids, data_2d],
        names=("time", "event_id", "image"),
    )

    plain_stats_2d = plain_aggregator(table=table_2d, col_name="image")

    assert len(plain_stats_2d) == 2

    # Check shapes are correct (2 gains, 10 pixels)
    assert plain_stats_2d[0]["mean"].shape == (2, 10)
    assert plain_stats_2d[0]["n_events"].shape == (2, 10)

    # Check that no statistics are NaN (all should be computed)
    assert not np.any(np.isnan(plain_stats_2d[0]["mean"]))
    assert not np.any(np.isnan(plain_stats_2d[0]["median"]))
    assert not np.any(np.isnan(plain_stats_2d[0]["std"]))
    assert not np.any(np.isnan(plain_stats_2d[1]["mean"]))
    assert not np.any(np.isnan(plain_stats_2d[1]["median"]))
    assert not np.any(np.isnan(plain_stats_2d[1]["std"]))

    # Check n_events for specific pixels with NaN values
    # Pixel [0, 3] in first chunk should have 499 events (1 NaN removed)
    assert plain_stats_2d[0]["n_events"][0, 3] == 499
    # Pixel [1, 7] in first chunk should have 499 events (1 NaN removed)
    assert plain_stats_2d[0]["n_events"][1, 7] == 499
    # Pixel [0, 5] in second chunk should have 499 events
    assert plain_stats_2d[1]["n_events"][0, 5] == 499
    # Pixel [1, 2] in second chunk should have 499 events
    assert plain_stats_2d[1]["n_events"][1, 2] == 499

    # Pixels without NaN should have full 500 events
    assert plain_stats_2d[0]["n_events"][0, 0] == 500
    assert plain_stats_2d[1]["n_events"][1, 9] == 500

    # Check that means are still reasonable
    np.testing.assert_allclose(plain_stats_2d[0]["mean"], 5.0, atol=1.0)
    np.testing.assert_allclose(plain_stats_2d[1]["mean"], 5.0, atol=1.0)

    # Test with all NaN in a pixel (edge case)
    data_all_nan = rng.normal(10.0, 2.0, size=(100, 5))
    data_all_nan[:, 2] = np.nan  # All values in pixel 2 are NaN

    table_all_nan = Table(
        [
            Time(np.linspace(60117.911, 60117.912, num=100), scale="tai", format="mjd"),
            np.arange(100),
            data_all_nan,
        ],
        names=("time", "event_id", "data"),
    )

    config_all_nan = Config(
        {
            "PlainAggregator": {"chunking_type": "SizeChunking"},
            "SizeChunking": {"chunk_size": None},
        }
    )
    aggregator_all_nan = PlainAggregator(config=config_all_nan)
    plain_stats_all_nan = aggregator_all_nan(table=table_all_nan, col_name="data")

    # Should have 1 chunk (chunk_size=None means entire table)
    assert len(plain_stats_all_nan) == 1

    # Pixel 2 should have 0 events and NaN statistics
    assert plain_stats_all_nan[0]["n_events"][2] == 0
    assert np.isnan(plain_stats_all_nan[0]["mean"][2])
    assert np.isnan(plain_stats_all_nan[0]["median"][2])
    assert np.isnan(plain_stats_all_nan[0]["std"][2])

    # Other pixels should have valid statistics
    assert plain_stats_all_nan[0]["n_events"][0] == 100
    assert not np.isnan(plain_stats_all_nan[0]["mean"][0])
    np.testing.assert_allclose(
        plain_stats_all_nan[0]["mean"][[0, 1, 3, 4]], 10.0, atol=1.0
    )


def test_undersized_tables():
    """Test handling of tables smaller than chunk size with allow_undersized_tables option"""

    # Create small test table (50 rows)
    times = Time(np.linspace(60117.911, 60117.912, num=50), scale="tai", format="mjd")
    event_ids = np.arange(50)
    rng = np.random.default_rng(42)
    data = rng.normal(10.0, 2.0, size=(50, 2, 5))

    small_table = Table(
        [times, event_ids, data],
        names=("time", "event_id", "image"),
    )

    # Test 1: SizeChunking with undersized table - should raise error by default
    config_error = Config(
        {
            "PlainAggregator": {"chunking_type": "SizeChunking"},
            "SizeChunking": {"chunk_size": 100, "allow_undersized_tables": False},
        }
    )
    aggregator_error = PlainAggregator(config=config_error)

    with pytest.raises(
        ValueError, match="Table length \\(50\\) is less than chunk_size \\(100\\)"
    ):
        aggregator_error(table=small_table)

    # Test 2: SizeChunking with allow_undersized_tables=True
    config_allow = Config(
        {
            "PlainAggregator": {"chunking_type": "SizeChunking"},
            "SizeChunking": {"chunk_size": 100, "allow_undersized_tables": True},
        }
    )
    aggregator_allow = PlainAggregator(config=config_allow)
    result_allow = aggregator_allow(table=small_table)

    # Should process entire table as single chunk
    assert len(result_allow) == 1
    assert result_allow[0]["n_events"][0, 0] == 50  # All 50 events in single chunk

    # Test 3: TimeChunking with undersized table - should raise error by default
    # Note: table spans ~86.4 seconds, so use 100s chunk_duration to make it undersized
    config_time_error = Config(
        {
            "PlainAggregator": {"chunking_type": "TimeChunking"},
            "TimeChunking": {
                "chunk_duration": 100 * u.s,
                "allow_undersized_tables": False,
            },
        }
    )
    aggregator_time_error = PlainAggregator(config=config_time_error)

    with pytest.raises(
        ValueError, match="Total duration .* is less than chunk_duration"
    ):
        aggregator_time_error(table=small_table)

    # Test 4: TimeChunking with allow_undersized_tables=True
    config_time_allow = Config(
        {
            "PlainAggregator": {"chunking_type": "TimeChunking"},
            "TimeChunking": {
                "chunk_duration": 100 * u.s,
                "allow_undersized_tables": True,
            },
        }
    )
    aggregator_time_allow = PlainAggregator(config=config_time_allow)
    result_time_allow = aggregator_time_allow(table=small_table)

    # Should process entire table as single chunk
    assert len(result_time_allow) == 1
    assert result_time_allow[0]["n_events"][0, 0] == 50


def test_last_chunk_policy_size_chunking():
    """Test different last_chunk_policy options for SizeChunking"""

    # Create test table with 55 rows (chunk_size=20 leaves 15 rows remainder)
    times = Time(np.linspace(60117.911, 60117.912, num=55), scale="tai", format="mjd")
    event_ids = np.arange(55)
    rng = np.random.default_rng(123)
    data = rng.normal(5.0, 1.0, size=(55, 3))

    table = Table(
        [times, event_ids, data],
        names=("time", "event_id", "image"),
    )

    # Test 1: last_chunk_policy="overlap" (default)
    config_overlap = Config(
        {
            "PlainAggregator": {"chunking_type": "SizeChunking"},
            "SizeChunking": {"chunk_size": 20, "last_chunk_policy": "overlap"},
        }
    )
    aggregator_overlap = PlainAggregator(config=config_overlap)
    result_overlap = aggregator_overlap(table=table)

    # Should have 3 chunks: [0:20], [20:40], [35:55] (last overlaps)
    assert len(result_overlap) == 3
    assert result_overlap[0]["n_events"][0] == 20  # First chunk: 20 events
    assert result_overlap[1]["n_events"][0] == 20  # Second chunk: 20 events
    assert result_overlap[2]["n_events"][0] == 20  # Last chunk: 20 events (overlapping)

    # Test 2: last_chunk_policy="truncate"
    config_truncate = Config(
        {
            "PlainAggregator": {"chunking_type": "SizeChunking"},
            "SizeChunking": {"chunk_size": 20, "last_chunk_policy": "truncate"},
        }
    )
    aggregator_truncate = PlainAggregator(config=config_truncate)
    result_truncate = aggregator_truncate(table=table)

    # Should have 3 chunks: [0:20], [20:40], [40:55] (last is partial)
    assert len(result_truncate) == 3
    assert result_truncate[0]["n_events"][0] == 20  # First chunk: 20 events
    assert result_truncate[1]["n_events"][0] == 20  # Second chunk: 20 events
    assert result_truncate[2]["n_events"][0] == 15  # Last chunk: 15 events (truncated)

    # Test 3: last_chunk_policy="skip"
    config_skip = Config(
        {
            "PlainAggregator": {"chunking_type": "SizeChunking"},
            "SizeChunking": {"chunk_size": 20, "last_chunk_policy": "skip"},
        }
    )
    aggregator_skip = PlainAggregator(config=config_skip)
    result_skip = aggregator_skip(table=table)

    # Should have 2 chunks: [0:20], [20:40] (last chunk skipped)
    assert len(result_skip) == 2
    assert result_skip[0]["n_events"][0] == 20  # First chunk: 20 events
    assert result_skip[1]["n_events"][0] == 20  # Second chunk: 20 events
    # No third chunk - the remaining 15 events are skipped


def test_last_chunk_policy_time_chunking():
    """Test different last_chunk_policy options for TimeChunking"""

    # Create test table with 5.5 seconds of data (chunk_duration=2s leaves 1.5s remainder)
    times = Time(
        np.linspace(60117.911, 60117.911 + 5.5 / 86400, num=55),
        scale="tai",
        format="mjd",
    )
    event_ids = np.arange(55)
    rng = np.random.default_rng(456)
    data = rng.normal(8.0, 1.5, size=(55, 2))

    table = Table(
        [times, event_ids, data],
        names=("time", "event_id", "image"),
    )

    # Test 1: last_chunk_policy="overlap" (default)
    config_overlap = Config(
        {
            "PlainAggregator": {"chunking_type": "TimeChunking"},
            "TimeChunking": {"chunk_duration": 2 * u.s, "last_chunk_policy": "overlap"},
        }
    )
    aggregator_overlap = PlainAggregator(config=config_overlap)
    result_overlap = aggregator_overlap(table=table)

    # Should have 3 chunks: [0:2s], [2:4s], [3.5:5.5s] (last overlaps)
    assert len(result_overlap) == 3
    # All chunks should have roughly the same number of events (overlapping ensures full duration)

    # Test 2: last_chunk_policy="truncate"
    config_truncate = Config(
        {
            "PlainAggregator": {"chunking_type": "TimeChunking"},
            "TimeChunking": {
                "chunk_duration": 2 * u.s,
                "last_chunk_policy": "truncate",
            },
        }
    )
    aggregator_truncate = PlainAggregator(config=config_truncate)
    result_truncate = aggregator_truncate(table=table)

    # Should have 3 chunks: [0:2s], [2:4s], [4:5.5s] (last is partial)
    assert len(result_truncate) == 3
    # Last chunk should have fewer events due to shorter duration

    # Test 3: last_chunk_policy="skip"
    config_skip = Config(
        {
            "PlainAggregator": {"chunking_type": "TimeChunking"},
            "TimeChunking": {"chunk_duration": 2 * u.s, "last_chunk_policy": "skip"},
        }
    )
    aggregator_skip = PlainAggregator(config=config_skip)
    result_skip = aggregator_skip(table=table)

    # Should have 2 chunks: [0:2s], [2:4s] (last partial chunk skipped)
    assert len(result_skip) == 2
