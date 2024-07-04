"""
Tests for StatisticsExtractor and related functions
"""

import numpy as np
from astropy.table import Table
from astropy.time import Time

from ctapipe.calib.camera.extractor import PlainExtractor, SigmaClippingExtractor


def test_extractors(example_subarray):
    """test basic functionality of the StatisticsExtractors"""

    # Create dummy data for testing
    times = Time(
        np.linspace(60117.911, 60117.9258, num=5000), scale="tai", format="mjd"
    )
    ped_dl1_data = np.random.normal(2.0, 5.0, size=(5000, 2, 1855))
    ff_charge_dl1_data = np.random.normal(77.0, 10.0, size=(5000, 2, 1855))
    ff_time_dl1_data = np.random.normal(18.0, 5.0, size=(5000, 2, 1855))
    # Create dl1 tables
    ped_dl1_table = Table(
        [times, ped_dl1_data],
        names=("time_mono", "image"),
    )
    ff_charge_dl1_table = Table(
        [times, ff_charge_dl1_data],
        names=("time_mono", "image"),
    )
    ff_time_dl1_table = Table(
        [times, ff_time_dl1_data],
        names=("time_mono", "peak_time"),
    )
    # Initialize the extractors
    ped_extractor = SigmaClippingExtractor(
        subarray=example_subarray, chunk_size=2500, outlier_method="standard_deviation"
    )
    ff_charge_extractor = SigmaClippingExtractor(
        subarray=example_subarray, chunk_size=2500, outlier_method="median"
    )
    ff_time_extractor = PlainExtractor(subarray=example_subarray, chunk_size=2500)

    # Extract the statistical values
    ped_stats_list = ped_extractor(dl1_table=ped_dl1_table)
    ff_charge_stats_list = ff_charge_extractor(dl1_table=ff_charge_dl1_table)
    ff_time_stats_list = ff_time_extractor(
        dl1_table=ff_time_dl1_table, col_name="peak_time"
    )
    # Check if the calculated statistical values are reasonable
    # for a camera with two gain channels
    assert not np.any(np.abs(ped_stats_list[0].mean - 2.0) > 1.5)
    assert not np.any(np.abs(ff_charge_stats_list[0].mean - 77.0) > 1.5)
    assert not np.any(np.abs(ff_time_stats_list[0].mean - 18.0) > 1.5)

    assert not np.any(np.abs(ped_stats_list[1].median - 2.0) > 1.5)
    assert not np.any(np.abs(ff_charge_stats_list[1].median - 77.0) > 1.5)
    assert not np.any(np.abs(ff_time_stats_list[1].median - 18.0) > 1.5)

    assert not np.any(np.abs(ped_stats_list[0].std - 5.0) > 1.5)
    assert not np.any(np.abs(ff_charge_stats_list[0].std - 10.0) > 1.5)
    assert not np.any(np.abs(ff_time_stats_list[0].std - 5.0) > 1.5)


def test_check_outliers(example_subarray):
    """test detection ability of outliers"""

    # Create dummy data for testing
    times = Time(
        np.linspace(60117.911, 60117.9258, num=5000), scale="tai", format="mjd"
    )
    ff_dl1_data = np.random.normal(77.0, 10.0, size=(5000, 2, 1855))
    # Insert outliers
    ff_dl1_data[:, 0, 120] = 120.0
    ff_dl1_data[:, 1, 67] = 120.0
    # Create dl1 table
    ff_dl1_table = Table(
        [times, ff_dl1_data],
        names=("time_mono", "image"),
    )
    # Initialize the extractor
    ff_charge_extractor = SigmaClippingExtractor(
        subarray=example_subarray, chunk_size=2500, outlier_method="median"
    )
    # Extract the statistical values
    ff_charge_stats_list = ff_charge_extractor(dl1_table=ff_dl1_table)
    # Check if outliers where detected correctly
    assert ff_charge_stats_list[0].median_outliers[0][120]
    assert ff_charge_stats_list[0].median_outliers[1][67]
    assert ff_charge_stats_list[1].median_outliers[0][120]
    assert ff_charge_stats_list[1].median_outliers[1][67]


def test_check_chunk_shift(example_subarray):
    """test the chunk shift option and the boundary case for the last chunk"""

    # Create dummy data for testing
    times = Time(
        np.linspace(60117.911, 60117.9258, num=5500), scale="tai", format="mjd"
    )
    ff_dl1_data = np.random.normal(77.0, 10.0, size=(5500, 2, 1855))
    # Create dl1 table
    ff_dl1_table = Table(
        [times, ff_dl1_data],
        names=("time_mono", "image"),
    )
    # Initialize the extractor
    ff_charge_extractor = SigmaClippingExtractor(
        subarray=example_subarray, chunk_size=2500, outlier_method="median"
    )
    # Extract the statistical values
    stats_list = ff_charge_extractor(dl1_table=ff_dl1_table)
    stats_list_chunk_shift = ff_charge_extractor(
        dl1_table=ff_dl1_table, chunk_shift=2000
    )
    # Check if three chunks are used for the extraction as the last chunk overflows
    assert len(stats_list) == 3
    # Check if two chunks are used for the extraction as the last chunk is dropped
    assert len(stats_list_chunk_shift) == 2
