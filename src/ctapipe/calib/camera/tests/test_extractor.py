"""
Tests for StatisticsExtractor and related functions
"""

import numpy as np
import pytest
from astropy.table import Table
from astropy.time import Time

from ctapipe.calib.camera.extractor import PlainExtractor, SigmaClippingExtractor


def test_extractors(example_subarray):
    """test basic functionality of the StatisticsExtractors"""

    # Create dummy data for testing
    times = Time(
        np.linspace(60117.911, 60117.9258, num=5000), scale="tai", format="mjd"
    )
    event_ids = np.linspace(35, 725000, num=5000, dtype=int)
    ped_data = np.random.normal(2.0, 5.0, size=(5000, 2, 1855))
    charge_data = np.random.normal(77.0, 10.0, size=(5000, 2, 1855))
    time_data = np.random.normal(18.0, 5.0, size=(5000, 2, 1855))
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
    # Initialize the extractors
    chunk_size = 2500
    ped_extractor = SigmaClippingExtractor(
        subarray=example_subarray, chunk_size=chunk_size
    )
    ff_charge_extractor = SigmaClippingExtractor(
        subarray=example_subarray, chunk_size=chunk_size
    )
    ff_time_extractor = PlainExtractor(subarray=example_subarray, chunk_size=chunk_size)

    # Extract the statistical values
    ped_stats = ped_extractor(table=ped_table)
    charge_stats = ff_charge_extractor(table=charge_table)
    time_stats = ff_time_extractor(table=time_table, col_name="peak_time")

    # Check if the start and end values are properly set for the timestamps and event IDs
    assert ped_stats[0]["time_start"] == times[0]
    assert time_stats[0]["event_id_start"] == event_ids[0]
    assert ped_stats[1]["time_end"] == times[-1]
    assert time_stats[1]["event_id_end"] == event_ids[-1]

    # Check if the calculated statistical values are reasonable
    # for a camera with two gain channels
    assert not np.any(np.abs(ped_stats[0]["mean"] - 2.0) > 1.5)
    assert not np.any(np.abs(charge_stats[0]["mean"] - 77.0) > 1.5)
    assert not np.any(np.abs(time_stats[0]["mean"] - 18.0) > 1.5)

    assert not np.any(np.abs(ped_stats[1]["median"] - 2.0) > 1.5)
    assert not np.any(np.abs(charge_stats[1]["median"] - 77.0) > 1.5)
    assert not np.any(np.abs(time_stats[1]["median"] - 18.0) > 1.5)

    assert not np.any(np.abs(ped_stats[0]["std"] - 5.0) > 1.5)
    assert not np.any(np.abs(charge_stats[0]["std"] - 10.0) > 1.5)
    assert not np.any(np.abs(time_stats[0]["std"] - 5.0) > 1.5)


def test_chunk_shift(example_subarray):
    """test the chunk shift option and the boundary case for the last chunk"""

    # Create dummy data for testing
    times = Time(
        np.linspace(60117.911, 60117.9258, num=5500), scale="tai", format="mjd"
    )
    event_ids = np.linspace(35, 725000, num=5500, dtype=int)
    charge_data = np.random.normal(77.0, 10.0, size=(5500, 2, 1855))
    # Create table
    charge_table = Table(
        [times, event_ids, charge_data],
        names=("time_mono", "event_id", "image"),
    )
    # Initialize the extractor
    extractor = SigmaClippingExtractor(subarray=example_subarray, chunk_size=2500)
    # Extract the statistical values
    chunk_stats = extractor(table=charge_table)
    chunk_stats_shift = extractor(table=charge_table, chunk_shift=2000)
    # Check if three chunks are used for the extraction as the last chunk overflows
    assert len(chunk_stats) == 3
    # Check if two chunks are used for the extraction as the last chunk is dropped
    assert len(chunk_stats_shift) == 2
    # Check if ValueError is raised when the chunk_size is larger than the length of table
    with pytest.raises(ValueError):
        _ = extractor(table=charge_table[1000:1500])
    # Check if ValueError is raised when the chunk_shift is smaller than the chunk_size
    with pytest.raises(ValueError):
        _ = extractor(table=charge_table, chunk_shift=3000)


def test_with_outliers(example_subarray):
    """test the robustness of the extractors in the presence of outliers"""

    # Create dummy data for testing
    times = Time(
        np.linspace(60117.911, 60117.9258, num=5000), scale="tai", format="mjd"
    )
    event_ids = np.linspace(35, 725000, num=5000, dtype=int)
    ped_data = np.random.normal(2.0, 5.0, size=(5000, 2, 1855))
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
    # Initialize the extractors
    sigmaclipping_extractor = SigmaClippingExtractor(
        subarray=example_subarray, chunk_size=2500
    )
    plain_extractor = PlainExtractor(subarray=example_subarray, chunk_size=2500)

    # Extract the statistical values
    sigmaclipping_chunk_stats = sigmaclipping_extractor(table=ped_table)
    plain_chunk_stats = plain_extractor(table=ped_table)

    # Check if SigmaClippingExtractor is robust to a few fake outliers as expected
    assert not np.any(np.abs(sigmaclipping_chunk_stats[0]["mean"] - 2.0) > 1.5)
    # Check if PlainExtractor is not robust to a few fake outliers as expected
    assert np.any(np.abs(plain_chunk_stats[0]["mean"] - 2.0) > 1.5)
