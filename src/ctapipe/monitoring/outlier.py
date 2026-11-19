"""
Outlier detection algorithms to identify faulty pixels
"""

from abc import abstractmethod
from typing import override

import numpy as np

from ..core import TelescopeComponent
from ..core.traits import Bool, Float, List

__all__ = [
    "MedianOutlierDetector",
    "OutlierDetector",
    "RangeOutlierDetector",
    "StdOutlierDetector",
]


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
            Column with chunk-wise aggregated statistic values (mean, median, or std)
            of shape (n_entries, n_channels, n_pixels), or histogram statistics of
            shape (n_entries, n_bins, n_channels, n_pixels).

        Returns
        -------
        boolean mask
            Mask of outliers with the same shape as ``column``.
        """


class RangeOutlierDetector(OutlierDetector):
    """
    Detect outliers based on a valid range.

    The clipping interval to set the thresholds for detecting outliers corresponds to
    a configurable range of valid statistic values.

    Either endpoint can be ``None`` to leave that side unbounded. Both endpoints
    are inclusive by default. To reject chunks with zero standard deviation,
    use ``validity_range=[0, None], min_inclusive=False`` and apply to ``std``.
    NaN values are not flagged as outliers.
    """

    validity_range = List(
        trait=Float(allow_none=True),
        default_value=[1.0, 2.0],
        help=(
            "Defines the range of acceptable values (lower, upper) in units of image values. "
            "Values outside this range will be flagged as outliers. "
            "Use None for an unbounded endpoint."
        ),
        minlen=2,
        maxlen=2,
    ).tag(config=True)

    min_inclusive = Bool(
        default_value=True,
        help="Whether the minimum value is acceptable.",
    ).tag(config=True)

    max_inclusive = Bool(
        default_value=True,
        help="Whether the maximum value is acceptable.",
    ).tag(config=True)

    @override
    def __call__(self, column):
        if column.ndim not in (3, 4):
            raise ValueError(
                f"Invalid shape of the column '{column.name}': '{column.shape}'. "
                "Expected statistic values of shape (n_entries, n_channels, n_pixels) "
                "or histogram values of shape (n_entries, n_bins, n_channels, n_pixels)."
            )

        outliers = np.zeros(column.shape, dtype=bool)
        lower, upper = self.validity_range
        if lower is not None:
            outliers |= column < lower if self.min_inclusive else column <= lower
        if upper is not None:
            outliers |= column > upper if self.max_inclusive else column >= upper
        return outliers


class MedianOutlierDetector(OutlierDetector):
    """
    Detect outliers based on the deviation from the camera median.

    The clipping interval to set the thresholds for detecting outliers is computed by multiplying
    the configurable factors and the camera median of the statistic values.
    """

    median_range_factors = List(
        trait=Float(),
        default_value=[-1.0, 1.0],
        help=(
            "Defines the range of acceptable values (lower, upper) as fractional deviations from the mean. "
            "Values outside of this range will be flagged as outliers."
        ),
        minlen=2,
        maxlen=2,
    ).tag(config=True)

    @override
    def __call__(self, column):
        # Accept both chunk-wise stats (3D) and histogram stats (4D)
        if column.ndim not in (3, 4):
            raise ValueError(
                f"Invalid shape of the column '{column.name}': '{column.shape}'. "
                "Expected statistic values of shape (n_entries, n_channels, n_pixels) "
                "or histogram values of shape (n_entries, n_bins, n_channels, n_pixels)."
            )
        pixel_axis = column.ndim - 1
        # Mask NaN values (e.g. from per-channel statistics of gain selected data)
        values = np.ma.masked_invalid(np.asarray(column))
        # Camera median
        camera_median = np.ma.median(values, axis=pixel_axis)
        # Detect outliers based on the deviation of the median distribution
        deviation = values - camera_median[..., np.newaxis]
        outliers = np.logical_or(
            deviation < self.median_range_factors[0] * camera_median[..., np.newaxis],
            deviation > self.median_range_factors[1] * camera_median[..., np.newaxis],
        )
        return np.ma.filled(outliers, False)


class StdOutlierDetector(OutlierDetector):
    """
    Detect outliers based on the deviation from the camera standard deviation.

    The clipping interval to set the thresholds for detecting outliers is computed by multiplying
    the configurable factors and the camera standard deviation of the statistic values.
    """

    std_range_factors = List(
        trait=Float(),
        default_value=[-1.0, 1.0],
        help=(
            "Defines the range of acceptable values (lower, upper) in units of standard deviations. "
            "Values outside of this range will be flagged as outliers."
        ),
        minlen=2,
        maxlen=2,
    ).tag(config=True)

    @override
    def __call__(self, column):
        # Accept both chunk-wise stats (3D) and histogram stats (4D)
        if column.ndim not in (3, 4):
            raise ValueError(
                f"Invalid shape of the column '{column.name}': '{column.shape}'. "
                "Expected statistic values of shape (n_entries, n_channels, n_pixels) "
                "or histogram values of shape (n_entries, n_bins, n_channels, n_pixels)."
            )
        pixel_axis = column.ndim - 1
        # Mask NaN values (e.g. from per-channel statistics of gain selected data)
        values = np.ma.masked_invalid(np.asarray(column))
        # Camera median
        camera_median = np.ma.median(values, axis=pixel_axis)
        # Camera std
        camera_std = np.ma.std(values, axis=pixel_axis)
        # Detect outliers based on the deviation of the standard deviation distribution
        deviation = values - camera_median[..., np.newaxis]
        outliers = np.logical_or(
            deviation < self.std_range_factors[0] * camera_std[..., np.newaxis],
            deviation > self.std_range_factors[1] * camera_std[..., np.newaxis],
        )
        return np.ma.filled(outliers, False)
