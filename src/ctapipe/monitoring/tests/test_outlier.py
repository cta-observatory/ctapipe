"""
Tests for OutlierDetector and related functions
"""

import numpy as np
from astropy.table import Table

from ctapipe.monitoring.outlier import (
    MedianOutlierDetector,
    RangeOutlierDetector,
    StdOutlierDetector,
)


def test_range_detection(example_subarray):
    """test the RangeOutlierDetector"""

    # Create dummy data for testing
    rng = np.random.default_rng(0)
    # Distribution mimics the mean values of peak times of flat-field events
    mean = rng.normal(18.0, 0.4, size=(50, 2, 1855))
    # Insert outliers
    mean[12, 0, 120] = 3.0
    mean[42, 1, 67] = 35.0
    # Create astropy table
    table = Table([mean], names=("mean",))
    # Initialize the outlier detector based on the range of valid values
    # In this test, the interval [15, 25] corresponds to the range (in waveform samples)
    # of accepted mean (or median) values of peak times of flat-field events
    detector = RangeOutlierDetector(
        subarray=example_subarray, validity_range=[15.0, 25.0]
    )
    # Detect outliers
    outliers = detector(table["mean"])
    # Construct the expected outliers
    expected_outliers = np.zeros((50, 2, 1855), dtype=bool)
    expected_outliers[12, 0, 120] = True
    expected_outliers[42, 1, 67] = True
    # Check if outliers where detected correctly
    np.testing.assert_array_equal(outliers, expected_outliers)


def test_median_detection(example_subarray):
    """test the MedianOutlierDetector"""

    # Create dummy data for testing
    rng = np.random.default_rng(0)
    # Distribution mimics the median values of charge images of flat-field events
    median = rng.normal(77.0, 0.6, size=(50, 2, 1855))
    # Insert outliers
    median[12, 0, 120] = 1.2
    median[42, 1, 67] = 21045.1
    # Create astropy table
    table = Table([median], names=("median",))
    # Initialize the outlier detector based on the deviation from the camera median
    # In this test, the interval [-0.9, 8] corresponds to multiplication factors
    # typical used for the median values of charge images of flat-field events
    detector = MedianOutlierDetector(
        subarray=example_subarray, median_range_factors=[-0.9, 8.0]
    )
    # Detect outliers
    outliers = detector(table["median"])
    # Construct the expected outliers
    expected_outliers = np.zeros((50, 2, 1855), dtype=bool)
    expected_outliers[12, 0, 120] = True
    expected_outliers[42, 1, 67] = True
    # Check if outliers where detected correctly
    np.testing.assert_array_equal(outliers, expected_outliers)


def test_std_detection(example_subarray):
    """test the StdOutlierDetector"""

    # Create dummy data for testing
    rng = np.random.default_rng(0)
    # Distribution mimics the std values of charge images of flat-field events
    ff_std = rng.normal(10.0, 2.0, size=(50, 2, 1855))
    # Distribution mimics the median values of charge images of pedestal events
    ped_median = rng.normal(2.0, 1.5, size=(50, 2, 1855))
    # Insert outliers
    ff_std[12, 0, 120] = 45.5
    ped_median[42, 1, 67] = 77.2
    # Create astropy table
    ff_table = Table([ff_std], names=("std",))
    ped_table = Table([ped_median], names=("median",))

    # Initialize the outlier detector based on the deviation from the camera standard deviation
    # In this test, the interval [-15, 15] corresponds to multiplication factors
    # typical used for the std values of charge images of flat-field events
    # and median (and std) values of charge images of pedestal events
    detector = StdOutlierDetector(
        subarray=example_subarray, std_range_factors=[-15.0, 15.0]
    )
    ff_outliers = detector(ff_table["std"])
    ped_outliers = detector(ped_table["median"])
    # Construct the expected outliers
    ff_expected_outliers = np.zeros((50, 2, 1855), dtype=bool)
    ff_expected_outliers[12, 0, 120] = True
    ped_expected_outliers = np.zeros((50, 2, 1855), dtype=bool)
    ped_expected_outliers[42, 1, 67] = True

    # Check if outliers where detected correctly
    np.testing.assert_array_equal(ff_outliers, ff_expected_outliers)
    np.testing.assert_array_equal(ped_outliers, ped_expected_outliers)
