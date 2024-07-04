"""
Tests for StatisticsExtractor and related functions
"""

import numpy as np
import pytest
from astropy.table import Table
from astropy.time import Time

from ctapipe.calib.camera.extractor import PlainExtractor, SigmaClippingExtractor


@pytest.fixture()
def plain_extractor(example_subarray):
    """test the PlainExtractor"""
    return PlainExtractor(subarray=example_subarray, chunk_size=2500)


@pytest.fixture()
def sigmaclipping_extractor(example_subarray):
    """test the SigmaClippingExtractor"""
    return SigmaClippingExtractor(subarray=example_subarray, chunk_size=2500)


def test_extractors(plain_extractor, sigmaclipping_extractor):
    """test basic functionality of the StatisticsExtractors"""

    # Create dummy data for testing
    times = Time(
        np.linspace(60117.911, 60117.9258, num=5000), scale="tai", format="mjd"
    )
    pedestal_dl1_data = np.random.normal(2.0, 5.0, size=(5000, 2, 1855))
    pedestal_event_type = np.full((5000,), 2)
    flatfield_charge_dl1_data = np.random.normal(77.0, 10.0, size=(5000, 2, 1855))
    flatfield_time_dl1_data = np.random.normal(18.0, 5.0, size=(5000, 2, 1855))
    flatfield_event_type = np.full((5000,), 0)
    # Create dl1 tables
    pedestal_dl1_table = Table(
        [times, pedestal_dl1_data, pedestal_event_type],
        names=("time_mono", "image", "event_type"),
    )
    flatfield_charge_dl1_table = Table(
        [times, flatfield_charge_dl1_data, flatfield_event_type],
        names=("time_mono", "image", "event_type"),
    )
    flatfield_time_dl1_table = Table(
        [times, flatfield_time_dl1_data, flatfield_event_type],
        names=("time_mono", "peak_time", "event_type"),
    )
    # Extract the statistical values
    pedestal_stats_list = sigmaclipping_extractor(dl1_table=pedestal_dl1_table)
    flatfield_charge_stats_list = sigmaclipping_extractor(
        dl1_table=flatfield_charge_dl1_table
    )
    flatfield_time_stats_list = plain_extractor(
        dl1_table=flatfield_time_dl1_table, col_name="peak_time"
    )
    # check if the calculated statistical values are reasonable
    # for a camera with two gain channels
    assert not np.any(np.abs(pedestal_stats_list[0].mean - 2.0) > 1.5)
    assert not np.any(np.abs(flatfield_charge_stats_list[0].mean - 77.0) > 1.5)
    assert not np.any(np.abs(flatfield_time_stats_list[0].mean - 18.0) > 1.5)

    assert not np.any(np.abs(pedestal_stats_list[1].median - 2.0) > 1.5)
    assert not np.any(np.abs(flatfield_charge_stats_list[1].median - 77.0) > 1.5)
    assert not np.any(np.abs(flatfield_time_stats_list[1].median - 18.0) > 1.5)

    assert not np.any(np.abs(pedestal_stats_list[0].std - 5.0) > 1.5)
    assert not np.any(np.abs(flatfield_charge_stats_list[0].std - 10.0) > 1.5)
    assert not np.any(np.abs(flatfield_time_stats_list[0].std - 5.0) > 1.5)


def test_check_outliers(sigmaclipping_extractor):
    """test detection ability of outliers"""

    # Create dummy data for testing
    times = Time(
        np.linspace(60117.911, 60117.9258, num=5000), scale="tai", format="mjd"
    )
    flatfield_dl1_data = np.random.normal(77.0, 10.0, size=(5000, 2, 1855))
    flatfield_event_type = np.full((5000,), 0)
    # insert outliers
    flatfield_dl1_data[:, 0, 120] = 120.0
    flatfield_dl1_data[:, 1, 67] = 120.0
    # Create dl1 table
    flatfield_dl1_table = Table(
        [times, flatfield_dl1_data, flatfield_event_type],
        names=("time_mono", "image", "event_type"),
    )
    # Extract the statistical values
    sigmaclipping_stats_list = sigmaclipping_extractor(dl1_table=flatfield_dl1_table)
    # check if outliers where detected correctly
    assert sigmaclipping_stats_list[0].median_outliers[0][120]
    assert sigmaclipping_stats_list[0].median_outliers[1][67]
    assert sigmaclipping_stats_list[1].median_outliers[0][120]
    assert sigmaclipping_stats_list[1].median_outliers[1][67]


def test_check_chunk_shift(sigmaclipping_extractor):
    """test the chunk shift option and the boundary case for the last chunk"""

    # Create dummy data for testing
    times = Time(
        np.linspace(60117.911, 60117.9258, num=5500), scale="tai", format="mjd"
    )
    flatfield_dl1_data = np.random.normal(77.0, 10.0, size=(5500, 2, 1855))
    flatfield_event_type = np.full((5500,), 0)
    # Create dl1 table
    flatfield_dl1_table = Table(
        [times, flatfield_dl1_data, flatfield_event_type],
        names=("time_mono", "image", "event_type"),
    )
    # Extract the statistical values
    stats_list = sigmaclipping_extractor(dl1_table=flatfield_dl1_table)
    stats_list_chunk_shift = sigmaclipping_extractor(
        dl1_table=flatfield_dl1_table, chunk_shift=2000
    )
    # check if three chunks are used for the extraction as the last chunk overflows
    assert len(stats_list) == 3
    # check if two chunks are used for the extraction as the last chunk is dropped
    assert len(stats_list_chunk_shift) == 2


def test_check_input(sigmaclipping_extractor):
    """test the input dl1 data"""

    # Create dummy data for testing
    times = Time(
        np.linspace(60117.911, 60117.9258, num=5000), scale="tai", format="mjd"
    )
    flatfield_dl1_data = np.random.normal(77.0, 10.0, size=(5000, 2, 1855))
    # Insert one event with wrong event type
    flatfield_event_type = np.full((5000,), 0)
    flatfield_event_type[0] = 2
    # Create dl1 table
    flatfield_dl1_table = Table(
        [times, flatfield_dl1_data, flatfield_event_type],
        names=("time_mono", "image", "event_type"),
    )
    # Try to extract the statistical values, which results in a ValueError
    with pytest.raises(ValueError):
        _ = sigmaclipping_extractor(dl1_table=flatfield_dl1_table)

    # Construct event_type column for cosmic events
    cosmic_event_type = np.full((5000,), 32)
    # Create dl1 table
    cosmic_dl1_table = Table(
        [times, flatfield_dl1_data, cosmic_event_type],
        names=("time_mono", "image", "event_type"),
    )
    # Try to extract the statistical values, which results in a ValueError
    with pytest.raises(ValueError):
        _ = sigmaclipping_extractor(dl1_table=cosmic_dl1_table)
