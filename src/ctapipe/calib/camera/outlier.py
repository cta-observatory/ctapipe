"""
Outlier detection algorithms to identify faulty pixels
"""

__all__ = [
    "OutlierDetection",
    "RangeBasedOutlierDetection",
    "MedianBasedOutlierDetection",
    "StdBasedOutlierDetection",
]

from abc import abstractmethod

import numpy as np

from ctapipe.core import TelescopeComponent
from ctapipe.core.traits import List


class OutlierDetection(TelescopeComponent):
    """
    Base class for outlier detection algorithms.
    """

    outliers_interval = List(
        [-1.0, 1.0],
        help=(
            "Interval of the multiplicative factor for detecting outliers based on"
            "the subcomponents."
        ),
    ).tag(config=True)

    @abstractmethod
    def __call__(self, column) -> bool:
        """
        Detect outliers in the provided table based on the specified column.

        This function should be implemented by subclasses to define the specific
        outlier detection approach. The function examines the statistics in the
        given column of the table and returns a boolean mask indicating which
        entries are considered outliers.

        Parameters
        ----------
        column : astropy.table.Column
            column with chunk-wise extracted statistics (mean, median, or std)
            of shape (n_entries, n_channels, n_pix)

        Returns
        -------
        boolean mask
            mask of outliers of shape (n_entries, n_channels, n_pix)
        """
        pass


class RangeBasedOutlierDetection(OutlierDetection):
    """
    Remove outliers based on a valid range.

    The interval `outliers_interval` corresponds to a range of valid statistical values.
    """

    def __call__(self, column):
        # Remove outliers is statistical values out a given range
        outliers = np.logical_or(
            column < self.outliers_interval[0],
            column > self.outliers_interval[1],
        )
        return outliers


class MedianBasedOutlierDetection(OutlierDetection):
    """
    Detect outliers based on the deviation from the camera median.

    The interval `outliers_interval` corresponds to the factors multiplied by the camera
    median of the provided statistical values to set the thresholds for identifying outliers.
    """

    def __call__(self, column):
        # Camera median
        camera_median = np.ma.median(column, axis=2)
        # Detect outliers based on the deviation of the median distribution
        deviation = column - camera_median[:, :, np.newaxis]
        outliers = np.logical_or(
            deviation < self.outliers_interval[0] * camera_median[:, :, np.newaxis],
            deviation > self.outliers_interval[1] * camera_median[:, :, np.newaxis],
        )
        return outliers


class StdBasedOutlierDetection(OutlierDetection):
    """
    Detect outliers based on the deviation from the camera standard deviation.

    The interval `outliers_interval` corresponds to the factors multiplied by the camera
    standard deviation of the provided statistical values to set the thresholds for identifying outliers.
    """

    def __call__(self, column):
        # Camera median
        camera_median = np.ma.median(column, axis=2)
        # Camera std
        camera_std = np.ma.std(column, axis=2)
        # Detect outliers based on the deviation of the standard deviation distribution
        deviation = column - camera_median[:, :, np.newaxis]
        outliers = np.logical_or(
            deviation < self.outliers_interval[0] * camera_std[:, :, np.newaxis],
            deviation > self.outliers_interval[1] * camera_std[:, :, np.newaxis],
        )
        return outliers
