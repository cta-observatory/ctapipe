"""
Extraction algorithms to compute the statistics from a sequence of images
"""

__all__ = [
    "StatisticsExtractor",
    "PlainExtractor",
    "SigmaClippingExtractor",
]

from abc import abstractmethod

import numpy as np
from astropy.stats import sigma_clipped_stats

from ctapipe.core import TelescopeComponent
from ctapipe.containers import StatisticsContainer
from ctapipe.core.traits import (
    Int,
    List,
)

class StatisticsExtractor(TelescopeComponent):
    """Base StatisticsExtractor component"""

    sample_size = Int(
        2500,
        help="Size of the sample used for the calculation of the statistical values",
    ).tag(config=True)
    image_median_cut_outliers = List(
        [-0.3, 0.3],
        help="""Interval of accepted image values \\
                (fraction with respect to camera median value)""",
    ).tag(config=True)
    image_std_cut_outliers = List(
        [-3, 3],
        help="""Interval (number of std) of accepted image standard deviation \\
                around camera median value""",
    ).tag(config=True)

    def __init__(self, subarray, config=None, parent=None, **kwargs):
        """
        Base component to handle the extraction of the statistics
        from a sequence of charges and pulse times (images).
>>>>>>> 58d868c8 (added stats extractor parent component)

        Parameters
        ----------
        kwargs
        """
        super().__init__(subarray=subarray, config=config, parent=parent, **kwargs)

    @abstractmethod
    def _extract(
        self, dl1_table, masked_pixels_of_sample=None, col_name="image"
    ) -> list:
        """
        Call the relevant functions to extract the statistics
        for the particular extractor.

        Parameters
        ----------
        dl1_table : ndarray
            dl1 table with images and times stored in a numpy array of shape
            (n_images, n_channels, n_pix).
        masked_pixels_of_sample : ndarray
            boolean array of masked pixels that are not available for processing
        col_name : string
            column name in the dl1 table

        Returns
        -------
        List StatisticsContainer:

            List of extracted statistics and validity ranges
        """


class PlainExtractor(StatisticsExtractor):
    """
    Extractor the statistics from a sequence of images
    using numpy and scipy functions
    """

    def _extract(
        self, dl1_table, masked_pixels_of_sample=None, col_name="image"
    ) -> list:

        # in python 3.12 itertools.batched can be used
        image_chunks = (
            dl1_table[col_name].data[i : i + self.sample_size]
            for i in range(0, len(dl1_table[col_name].data), self.sample_size)
        )
        time_chunks = (
            dl1_table["time"][i : i + self.sample_size]
            for i in range(0, len(dl1_table["time"]), self.sample_size)
        )

        # Calculate the statistics from a sequence of images
        stats_list = []
        for images, times in zip(image_chunks, time_chunks):
            stats_list.append(
                self._plain_extraction(images, times, masked_pixels_of_sample)
            )
        return stats_list

    def _plain_extraction(
        self, images, times, masked_pixels_of_sample
    ) -> StatisticsContainer:

        # ensure numpy array
        masked_images = np.ma.array(images, mask=masked_pixels_of_sample)

        # median over the sample per pixel
        pixel_median = np.ma.median(masked_images, axis=0)

        # mean over the sample per pixel
        pixel_mean = np.ma.mean(masked_images, axis=0)

        # std over the sample per pixel
        pixel_std = np.ma.std(masked_images, axis=0)

        # outliers from median
        image_median_outliers = np.logical_or(
            pixel_median < self.image_median_cut_outliers[0],
            pixel_median > self.image_median_cut_outliers[1],
        )

        return StatisticsContainer(
            validity_start=times[0],
            validity_stop=times[-1],
            mean=pixel_mean.filled(np.nan),
            median=pixel_median.filled(np.nan),
            median_outliers=image_median_outliers.filled(True),
            std=pixel_std.filled(np.nan),
        )

class SigmaClippingExtractor(StatisticsExtractor):
    """
    Extracts the statistics from a sequence of images
    using astropy's sigma clipping functions
    """

    sigma_clipping_max_sigma = Int(
        default_value=4,
        help="Maximal value for the sigma clipping outlier removal",
    ).tag(config=True)
    sigma_clipping_iterations = Int(
        default_value=5,
        help="Number of iterations for the sigma clipping outlier removal",
    ).tag(config=True)

    def _extract(
        self, dl1_table, masked_pixels_of_sample=None, col_name="image"
    ) -> list:

        # in python 3.12 itertools.batched can be used
        image_chunks = (
            dl1_table[col_name].data[i : i + self.sample_size]
            for i in range(0, len(dl1_table[col_name].data), self.sample_size)
        )
        time_chunks = (
            dl1_table["time"][i : i + self.sample_size]
            for i in range(0, len(dl1_table["time"]), self.sample_size)
        )

        # Calculate the statistics from a sequence of images
        stats_list = []
        for images, times in zip(image_chunks, time_chunks):
            stats_list.append(
                self._sigmaclipping_extraction(images, times, masked_pixels_of_sample)
            )
        return stats_list

    def _sigmaclipping_extraction(
        self, images, times, masked_pixels_of_sample
    ) -> StatisticsContainer:

        # ensure numpy array
        masked_images = np.ma.array(images, mask=masked_pixels_of_sample)

        # mean, median, and std over the sample per pixel
        max_sigma = self.sigma_clipping_max_sigma
        pixel_mean, pixel_median, pixel_std = sigma_clipped_stats(
            masked_images,
            sigma=max_sigma,
            maxiters=self.sigma_clipping_iterations,
            cenfunc="mean",
            axis=0,
        )

        # mask pixels without defined statistical values
        pixel_mean = np.ma.array(pixel_mean, mask=np.isnan(pixel_mean))
        pixel_median = np.ma.array(pixel_median, mask=np.isnan(pixel_median))
        pixel_std = np.ma.array(pixel_std, mask=np.isnan(pixel_std))

        unused_values = np.abs(masked_images - pixel_mean) > (max_sigma * pixel_std)

        # add outliers identified by sigma clipping for following operations
        masked_images.mask |= unused_values

        # median of the median over the camera
        median_of_pixel_median = np.ma.median(pixel_median, axis=1)

        # median of the std over the camera
        median_of_pixel_std = np.ma.median(pixel_std, axis=1)

        # std of the std over camera
        std_of_pixel_std = np.ma.std(pixel_std, axis=1)

        # outliers from median
        image_deviation = pixel_median - median_of_pixel_median[:, np.newaxis]
        image_median_outliers = np.logical_or(
            image_deviation
            < self.image_median_cut_outliers[0] # pylint: disable=unsubscriptable-object
            * median_of_pixel_median[
                :, np.newaxis
            ],
            image_deviation
            > self.image_median_cut_outliers[1] # pylint: disable=unsubscriptable-object
            * median_of_pixel_median[
                :, np.newaxis
            ],
        )

        # outliers from standard deviation
        deviation = pixel_std - median_of_pixel_std[:, np.newaxis]
        image_std_outliers = np.logical_or(
            deviation
            < self.image_std_cut_outliers[0] # pylint: disable=unsubscriptable-object
            * std_of_pixel_std[:, np.newaxis],
            deviation
            > self.image_std_cut_outliers[1] # pylint: disable=unsubscriptable-object
            * std_of_pixel_std[:, np.newaxis],
        )

        return StatisticsContainer(
            validity_start=times[0],
            validity_stop=times[-1],
            mean=pixel_mean.filled(np.nan),
            median=pixel_median.filled(np.nan),
            median_outliers=image_median_outliers.filled(True),
            std=pixel_std.filled(np.nan),
            std_outliers=image_std_outliers.filled(True),
        )
