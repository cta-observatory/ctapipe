"""
Outlier detection algorithms to identify faulty pixels
"""

__all__ = [
    "OutlierDetector",
    "RangeOutlierDetector",
    "MedianOutlierDetector",
    "StdOutlierDetector",
]

from abc import abstractmethod

import numpy as np

from ctapipe.core import TelescopeComponent
from ctapipe.core.traits import List


class OutlierDetector(TelescopeComponent):
    """
    Base class for outlier detection algorithms.
    """

    @abstractmethod
    def __call__(self, column) -> bool:
        """
        Detect outliers in the provided column.

        This function should be implemented by subclasses to define the specific
        outlier detection approach. The function examines the statistics in the
        given column of the table and returns a boolean mask indicating which
        entries are considered as outliers.

        Parameters
        ----------
        column : astropy.table.Column
            column with chunk-wise aggregated statistic values (mean, median, or std)
            of shape (n_entries, n_channels, n_pix)

        Returns
        -------
        boolean mask
            mask of outliers of shape (n_entries, n_channels, n_pix)
        """
        pass


class RangeOutlierDetector(OutlierDetector):
    """
    Detect outliers based on a valid range.

    The clipping interval to set the thresholds for detecting outliers corresponds to
    a configurable range of valid statistic values.
    """

    validity_range = List(
        [1.0, 2.0],
        help=(
            "Range of valid statistic values (in units of the image value)."
            "Values outside the range are identified as outliers."
        ),
    ).tag(config=True)

    def __call__(self, column):
        # Remove outliers is statistical values out a given range
        outliers = np.logical_or(
            column < self.validity_range[0],
            column > self.validity_range[1],
        )
        return outliers


class MedianOutlierDetector(OutlierDetector):
    """
    Detect outliers based on the deviation from the camera median.

    The clipping interval to set the thresholds for detecting outliers is computed by multiplying
    the configurable factors and the camera median of the statistic values.
    """

    median_range_factors = List(
        [-1.0, 1.0],
        help=(
            "Multiplicative factors (unitless) applied to the camera median"
            "of the provided statistic values to define a valid range based on the"
            "deviation of the values to its camera median."
        ),
    ).tag(config=True)

    def __call__(self, column):
        # Camera median
        camera_median = np.ma.median(column, axis=2)
        # Detect outliers based on the deviation of the median distribution
        deviation = column - camera_median[:, :, np.newaxis]
        outliers = np.logical_or(
            deviation < self.median_range_factors[0] * camera_median[:, :, np.newaxis],
            deviation > self.median_range_factors[1] * camera_median[:, :, np.newaxis],
        )
        return outliers


class StdOutlierDetector(OutlierDetector):
    """
    Detect outliers based on the deviation from the camera standard deviation.

    The clipping interval to set the thresholds for detecting outliers is computed by multiplying
    the configurable factors and the camera standard deviation of the statistic values.
    """

    std_range_factors = List(
        [-1.0, 1.0],
        help=(
            "Multiplicative factors (unitless) applied to the camera standard deviation"
            "of the provided statistic values to define a valid range based on the"
            "deviation of the values to its camera median."
        ),
    ).tag(config=True)

    def __call__(self, column):
        # Camera median
        camera_median = np.ma.median(column, axis=2)
        # Camera std
        camera_std = np.ma.std(column, axis=2)
        # Detect outliers based on the deviation of the standard deviation distribution
        deviation = column - camera_median[:, :, np.newaxis]
        outliers = np.logical_or(
            deviation < self.std_range_factors[0] * camera_std[:, :, np.newaxis],
            deviation > self.std_range_factors[1] * camera_std[:, :, np.newaxis],
        )
        return outliers
