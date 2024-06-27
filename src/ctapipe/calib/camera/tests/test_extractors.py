"""
Tests for StatisticsExtractor and related functions
"""

import numpy as np
import pytest
from astropy.table import Table
from astropy.time import Time

from ctapipe.calib.camera.extractor import PlainExtractor, SigmaClippingExtractor


@pytest.fixture(name="test_plainextractor")
def fixture_test_plainextractor(example_subarray):
    """test the PlainExtractor"""
    return PlainExtractor(subarray=example_subarray, chunk_size=2500)


@pytest.fixture(name="test_sigmaclippingextractor")
def fixture_test_sigmaclippingextractor(example_subarray):
    """test the SigmaClippingExtractor"""
    return SigmaClippingExtractor(subarray=example_subarray, chunk_size=2500)


def test_extractors(test_plainextractor, test_sigmaclippingextractor):
    """test basic functionality of the StatisticsExtractors"""

    times = Time(
        np.linspace(60117.911, 60117.9258, num=5000), scale="tai", format="mjd"
    )
    pedestal_dl1_data = np.random.normal(2.0, 5.0, size=(5000, 2, 1855))
    pedestal_event_type = np.full((5000,), 2)
    flatfield_dl1_data = np.random.normal(77.0, 10.0, size=(5000, 2, 1855))
    flatfield_event_type = np.full((5000,), 0)

    pedestal_dl1_table = Table(
        [times, pedestal_dl1_data, pedestal_event_type],
        names=("time_mono", "image", "event_type"),
    )
    flatfield_dl1_table = Table(
        [times, flatfield_dl1_data, flatfield_event_type],
        names=("time_mono", "image", "event_type"),
    )

    plain_stats_list = test_plainextractor(dl1_table=pedestal_dl1_table)
    sigmaclipping_stats_list = test_sigmaclippingextractor(
        dl1_table=flatfield_dl1_table
    )

    assert not np.any(np.abs(plain_stats_list[0].mean - 2.0) > 1.5)
    assert not np.any(np.abs(sigmaclipping_stats_list[0].mean - 77.0) > 1.5)

    assert not np.any(np.abs(plain_stats_list[0].mean - 2.0) > 1.5)
    assert not np.any(np.abs(sigmaclipping_stats_list[0].mean - 77.0) > 1.5)

    assert not np.any(np.abs(plain_stats_list[1].median - 2.0) > 1.5)
    assert not np.any(np.abs(sigmaclipping_stats_list[1].median - 77.0) > 1.5)

    assert not np.any(np.abs(plain_stats_list[0].std - 5.0) > 1.5)
    assert not np.any(np.abs(sigmaclipping_stats_list[0].std - 10.0) > 1.5)


def test_check_outliers(test_sigmaclippingextractor):
    """test detection ability of outliers"""

    times = Time(
        np.linspace(60117.911, 60117.9258, num=5000), scale="tai", format="mjd"
    )
    flatfield_dl1_data = np.random.normal(77.0, 10.0, size=(5000, 2, 1855))
    flatfield_event_type = np.full((5000,), 0)
    # insert outliers
    flatfield_dl1_data[:, 0, 120] = 120.0
    flatfield_dl1_data[:, 1, 67] = 120.0
    flatfield_dl1_table = Table(
        [times, flatfield_dl1_data, flatfield_event_type],
        names=("time_mono", "image", "event_type"),
    )
    sigmaclipping_stats_list = test_sigmaclippingextractor(
        dl1_table=flatfield_dl1_table
    )

    # check if outliers where detected correctly
    assert sigmaclipping_stats_list[0].median_outliers[0][120]
    assert sigmaclipping_stats_list[0].median_outliers[1][67]
    assert sigmaclipping_stats_list[1].median_outliers[0][120]
    assert sigmaclipping_stats_list[1].median_outliers[1][67]


def test_check_chunk_shift(test_sigmaclippingextractor):
    """test the chunk shift option and the boundary case for the last chunk"""

    times = Time(
        np.linspace(60117.911, 60117.9258, num=5000), scale="tai", format="mjd"
    )
    flatfield_dl1_data = np.random.normal(77.0, 10.0, size=(5000, 2, 1855))
    flatfield_event_type = np.full((5000,), 0)
    # insert outliers
    flatfield_dl1_table = Table(
        [times, flatfield_dl1_data, flatfield_event_type],
        names=("time_mono", "image", "event_type"),
    )
    sigmaclipping_stats_list = test_sigmaclippingextractor(
        dl1_table=flatfield_dl1_table, chunk_shift=2000
    )

    # check if three chunks are used for the extraction
    assert len(sigmaclipping_stats_list) == 3
