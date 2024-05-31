"""
Tests for StatisticsExtractor and related functions
"""

from astropy.table import QTable
import numpy as np

from ctapipe.calib.camera.extractor import PlainExtractor, SigmaClippingExtractor


def test_extractors(example_subarray):
    """test basic functionality of the StatisticsExtractors"""

    plain_extractor = PlainExtractor(subarray=example_subarray, sample_size=2500)
    sigmaclipping_extractor = SigmaClippingExtractor(
        subarray=example_subarray, sample_size=2500
    )
    times = np.linspace(60117.911, 60117.9258, num=5000)
    pedestal_dl1_data = np.random.normal(2.0, 5.0, size=(5000, 2, 1855))
    flatfield_dl1_data = np.random.normal(77.0, 10.0, size=(5000, 2, 1855))

    pedestal_dl1_table = QTable([times, pedestal_dl1_data], names=("time", "image"))
    flatfield_dl1_table = QTable([times, flatfield_dl1_data], names=("time", "image"))

    plain_stats_list = plain_extractor(dl1_table=pedestal_dl1_table)
    sigmaclipping_stats_list = sigmaclipping_extractor(dl1_table=flatfield_dl1_table)

    assert np.any(np.abs(plain_stats_list[0].mean - 2.0) > 1.5) is False
    assert np.any(np.abs(sigmaclipping_stats_list[0].mean - 77.0) > 1.5) is False

    assert np.any(np.abs(plain_stats_list[0].mean - 2.0) > 1.5) is False
    assert np.any(np.abs(sigmaclipping_stats_list[0].mean - 77.0) > 1.5) is False

    assert np.any(np.abs(plain_stats_list[1].median - 2.0) > 1.5) is False
    assert np.any(np.abs(sigmaclipping_stats_list[1].median - 77.0) > 1.5) is False

    assert np.any(np.abs(plain_stats_list[0].std - 5.0) > 1.5) is False
    assert np.any(np.abs(sigmaclipping_stats_list[0].std - 10.0) > 1.5) is False


def test_check_outliers(example_subarray):
    """test detection ability of outliers"""

    sigmaclipping_extractor = SigmaClippingExtractor(
        subarray=example_subarray, sample_size=2500
    )
    times = np.linspace(60117.911, 60117.9258, num=5000)
    flatfield_dl1_data = np.random.normal(77.0, 10.0, size=(5000, 2, 1855))
    # insert outliers
    flatfield_dl1_data[:, 0, 120] = 120.0
    flatfield_dl1_data[:, 1, 67] = 120.0
    flatfield_dl1_table = QTable([times, flatfield_dl1_data], names=("time", "image"))
    sigmaclipping_stats_list = sigmaclipping_extractor(dl1_table=flatfield_dl1_table)

    # check if outliers where detected correctly
    assert sigmaclipping_stats_list[0].median_outliers[0][120] is True
    assert sigmaclipping_stats_list[0].median_outliers[1][67] is True
    assert sigmaclipping_stats_list[1].median_outliers[0][120] is True
    assert sigmaclipping_stats_list[1].median_outliers[1][67] is True
