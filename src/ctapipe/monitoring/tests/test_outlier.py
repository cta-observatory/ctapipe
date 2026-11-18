"""
Tests for OutlierDetector and related functions
"""

import numpy as np
import pytest
from astropy.table import Table

from ctapipe.monitoring.outlier import (
    MedianOutlierDetector,
    OutlierDetector,
    RangeOutlierDetector,
    StdOutlierDetector,
)


def test_range_detection(example_subarray):
    """Test the RangeOutlierDetector"""
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


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize(
    ("min_inclusive", "max_inclusive", "expected"),
    [
        (True, True, [True, False, False, False, True, False]),
        (True, False, [True, False, False, True, True, False]),
        (False, True, [True, True, False, False, True, False]),
        (False, False, [True, True, False, True, True, False]),
    ],
)
def test_range_boundaries(
    example_subarray, ndim, min_inclusive, max_inclusive, expected
):
    values = np.array([-1, 0, 1, 2, 3, np.nan]).reshape((1,) * (ndim - 1) + (6,))
    table = Table([values], names=["values"])
    detector = RangeOutlierDetector(
        subarray=example_subarray,
        validity_range=[0, 2],
        min_inclusive=min_inclusive,
        max_inclusive=max_inclusive,
    )

    np.testing.assert_array_equal(
        detector(table["values"]), np.array(expected).reshape(values.shape)
    )


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"validity_range": [None, None]}, [False, False, False, False, False, False]),
        (
            {
                "validity_range": [0, None],
                "min_inclusive": False,
                "max_inclusive": True,
            },
            [True, True, True, False, False, False],
        ),
        ({"validity_range": [0, None]}, [True, True, False, False, False, False]),
        ({"validity_range": [None, 0]}, [False, False, False, True, True, False]),
        (
            {"validity_range": [None, 0], "max_inclusive": False},
            [False, False, True, True, True, False],
        ),
    ],
)
def test_range_optional_bounds(example_subarray, ndim, config, expected):
    values = np.array([-np.inf, -1, 0, 1, np.inf, np.nan]).reshape(
        (1,) * (ndim - 1) + (6,)
    )
    table = Table([values], names=["std"])
    detector = OutlierDetector.from_name(
        "RangeOutlierDetector", subarray=example_subarray, **config
    )

    np.testing.assert_array_equal(
        detector(table["std"]), np.array(expected).reshape(values.shape)
    )


def test_median_detection(example_subarray):
    """Test the MedianOutlierDetector"""
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
    """Test the StdOutlierDetector"""
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


def test_check_for_column_shape(example_subarray):
    """Test the check for the shape of the column"""
    # Create dummy data for testing
    rng = np.random.default_rng(0)
    # Distribution mimics the median values of gain-selected charge images of flat-field events
    median = rng.normal(77.0, 0.6, size=(50, 1855))
    # Create astropy table
    table = Table([median], names=("median",))
    # Initialize the outlier detector based on the deviation from the camera median
    outlier_detectors = [
        "RangeOutlierDetector",
        "MedianOutlierDetector",
        "StdOutlierDetector",
    ]
    for outlier_detector_name in outlier_detectors:
        outlier_detector = OutlierDetector.from_name(
            outlier_detector_name,
            subarray=example_subarray,
        )
        # Check if ValueError is raised when the provided column shape is invalid
        with pytest.raises(ValueError, match="Invalid shape of the column"):
            outlier_detector(table["median"])


@pytest.mark.parametrize(
    ("detector_cls", "config"),
    [
        (MedianOutlierDetector, {"median_range_factors": [-0.9, 8.0]}),
        (StdOutlierDetector, {"std_range_factors": [-15.0, 15.0]}),
    ],
)
def test_detection_with_nan(example_subarray, detector_cls, config):
    """Test that NaN values are ignored and not flagged as outliers"""
    rng = np.random.default_rng(0)
    values = rng.normal(77.0, 0.6, size=(10, 2, 1855))
    # Mimic per-channel statistics of gain selected data with mostly empty low gain
    values[:, 1, 1000:] = np.nan
    values[3, 1, 50] = 21045.1
    values[5, 0, 120] = 21045.1
    table = Table([values], names=("values",))

    detector = detector_cls(subarray=example_subarray, **config)
    outliers = detector(table["values"])

    expected_outliers = np.zeros((10, 2, 1855), dtype=bool)
    expected_outliers[3, 1, 50] = True
    expected_outliers[5, 0, 120] = True
    np.testing.assert_array_equal(outliers, expected_outliers)
