"""
Tests for StatisticsExtractor and related functions
"""

from astropy.table import QTable
import numpy as np
import pytest
from ctapipe.calib.camera.extractor import PlainExtractor, SigmaClippingExtractor

@pytest.fixture(name="test_plainextractor")
def fixture_test_plainextractor(example_subarray):
    """test the PlainExtractor"""
    return PlainExtractor(
        subarray=example_subarray, chunk_size=2500
    )

@pytest.fixture(name="test_sigmaclippingextractor")
def fixture_test_sigmaclippingextractor(example_subarray):
    """test the SigmaClippingExtractor"""
    return SigmaClippingExtractor(
        subarray=example_subarray, chunk_size=2500
    )

def test_extractors(test_plainextractor, test_sigmaclippingextractor):
    """test basic functionality of the StatisticsExtractors"""

    times = np.linspace(60117.911, 60117.9258, num=5000)
    pedestal_dl1_data = np.random.normal(2.0, 5.0, size=(5000, 2, 1855))
    flatfield_dl1_data = np.random.normal(77.0, 10.0, size=(5000, 2, 1855))

    pedestal_dl1_table = QTable([times, pedestal_dl1_data], names=("time", "image"))
    flatfield_dl1_table = QTable([times, flatfield_dl1_data], names=("time", "image"))

    plain_stats_list = test_plainextractor(dl1_table=pedestal_dl1_table)
    sigmaclipping_stats_list = test_sigmaclippingextractor(
        dl1_table=flatfield_dl1_table
    )

    assert np.any(np.abs(plain_stats_list[0].mean - 2.0) > 1.5) is False
    assert np.any(np.abs(sigmaclipping_stats_list[0].mean - 77.0) > 1.5) is False

    assert np.any(np.abs(plain_stats_list[0].mean - 2.0) > 1.5) is False
    assert np.any(np.abs(sigmaclipping_stats_list[0].mean - 77.0) > 1.5) is False

    assert np.any(np.abs(plain_stats_list[1].median - 2.0) > 1.5) is False
    assert np.any(np.abs(sigmaclipping_stats_list[1].median - 77.0) > 1.5) is False

    assert np.any(np.abs(plain_stats_list[0].std - 5.0) > 1.5) is False
    assert np.any(np.abs(sigmaclipping_stats_list[0].std - 10.0) > 1.5) is False


def test_check_outliers(test_sigmaclippingextractor):
    """test detection ability of outliers"""

    times = np.linspace(60117.911, 60117.9258, num=5000)
    flatfield_dl1_data = np.random.normal(77.0, 10.0, size=(5000, 2, 1855))
    # insert outliers
    flatfield_dl1_data[:, 0, 120] = 120.0
    flatfield_dl1_data[:, 1, 67] = 120.0
    flatfield_dl1_table = QTable([times, flatfield_dl1_data], names=("time", "image"))
    sigmaclipping_stats_list = test_sigmaclippingextractor(
        dl1_table=flatfield_dl1_table
    )

    # check if outliers where detected correctly
    assert sigmaclipping_stats_list[0].median_outliers[0][120] is True
    assert sigmaclipping_stats_list[0].median_outliers[1][67] is True
    assert sigmaclipping_stats_list[1].median_outliers[0][120] is True
    assert sigmaclipping_stats_list[1].median_outliers[1][67] is True
    

def test_check_chunk_shift(test_sigmaclippingextractor):
    """test the chunk shift option and the boundary case for the last chunk"""

    times = np.linspace(60117.911, 60117.9258, num=5000)
    flatfield_dl1_data = np.random.normal(77.0, 10.0, size=(5000, 2, 1855))
    # insert outliers
    flatfield_dl1_table = QTable([times, flatfield_dl1_data], names=("time", "image"))
    sigmaclipping_stats_list = test_sigmaclippingextractor(
        dl1_table=flatfield_dl1_table,
        chunk_shift=2000
    )

    # check if three chunks are used for the extraction
    assert len(sigmaclipping_stats_list) == 3

