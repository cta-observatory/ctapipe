"""
Extraction algorithms to compute the statistics from a chunk of images
"""

__all__ = [
    "StatisticsExtractor",
    "PlainExtractor",
    "SigmaClippingExtractor",
]

from abc import abstractmethod

import numpy as np
from astropy.stats import sigma_clipped_stats

from ctapipe.containers import StatisticsContainer
from ctapipe.core import TelescopeComponent
from ctapipe.core.traits import (
    Int,
    List,
)


class StatisticsExtractor(TelescopeComponent):
    """Base StatisticsExtractor component"""

    chunk_size = Int(
        2500,
        help="Size of the chunk used for the calculation of the statistical values",
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
        from a chunk of charges and pulse times (images).

        Parameters
        ----------
        kwargs
        """
        super().__init__(subarray=subarray, config=config, parent=parent, **kwargs)

    def __call__(
        self,
        dl1_table,
        masked_pixels_of_sample=None,
        chunk_shift=None,
        col_name="image",
    ) -> list:
        """
        Call the relevant functions to extract the statistics
        for the particular extractor.

        Parameters
        ----------
        dl1_table : astropy.table.Table
            DL1 table with images of shape (n_images, n_channels, n_pix),
            timestamps of shape (n_images, ), and event type (n_images, )
            stored in an astropy Table
        masked_pixels_of_sample : ndarray
            boolean array of masked pixels of shape (n_pix, ) that are not available for processing
        chunk_shift : int
            number of samples to shift the extraction chunk
        col_name : string
            column name in the DL1 table

        Returns
        -------
        List StatisticsContainer:
            List of extracted statistics and extraction chunks
        """

        # Check if the provided data contains an unique type of calibration events
        # and if the events contain signal or noise.
        if np.all(dl1_table["event_type"] == 0) or np.all(dl1_table["event_type"] == 1):
            outlier_check = "SIGNAL"
        elif (
            np.all(dl1_table["event_type"] == 2)
            or np.all(dl1_table["event_type"] == 3)
            or np.all(dl1_table["event_type"] == 4)
        ):
            outlier_check = "NOISE"
        else:
            raise ValueError(
                "Invalid input data. Only dl1-like images of claibration events are accepted!"
            )

        # Check if the length of the dl1 table is greater or equal than the size of the chunk.
        if len(dl1_table[col_name]) < self.chunk_size:
            raise ValueError(
                f"The length of the DL1 table ({len(dl1_table[col_name])}) must be greater or equal than the size of the chunk ({self.chunk_size})."
            )
        # If no chunk_shift is provided, the chunk_shift is set to self.chunk_size
        # meaning that the extraction chunks are not overlapping.
        if chunk_shift is None:
            chunk_shift = self.chunk_size

        # Function to split table data into appropriated chunks
        def _get_chunks(dl1_table_data):
            return [
                (
                    dl1_table_data[i : i + self.chunk_size]
                    if i + self.chunk_size <= len(dl1_table_data)
                    else dl1_table_data[
                        len(dl1_table_data) - self.chunk_size : len(dl1_table_data)
                    ]
                )
                for i in range(0, len(dl1_table_data), chunk_shift)
            ]

        # Get the chunks for the timestamps and selected column name
        time_chunks = _get_chunks(dl1_table["time_mono"])
        image_chunks = _get_chunks(dl1_table[col_name].data)

        # Calculate the statistics from a chunk of images
        stats_list = []
        for images, times in zip(image_chunks, time_chunks):
            stats_list.append(
                self._extract(images, times, masked_pixels_of_sample, outlier_check)
            )
        return stats_list

    @abstractmethod
    def _extract(
        self, images, times, masked_pixels_of_sample, outlier_check
    ) -> StatisticsContainer:
        pass


class PlainExtractor(StatisticsExtractor):
    """
    Extract the statistics from a chunk of images
    using numpy and scipy functions
    """

    def _extract(
        self, images, times, masked_pixels_of_sample, outlier_check
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
            extraction_start=times[0],
            extraction_stop=times[-1],
            mean=pixel_mean.filled(np.nan),
            median=pixel_median.filled(np.nan),
            median_outliers=image_median_outliers.filled(True),
            std=pixel_std.filled(np.nan),
            std_outliers=np.full(np.shape(image_median_outliers), False),
        )


class SigmaClippingExtractor(StatisticsExtractor):
    """
    Extract the statistics from a chunk of images
    using astropy's sigma clipping functions
    """

    max_sigma = Int(
        default_value=4,
        help="Maximal value for the sigma clipping outlier removal",
    ).tag(config=True)
    iterations = Int(
        default_value=5,
        help="Number of iterations for the sigma clipping outlier removal",
    ).tag(config=True)

    def _extract(
        self, images, times, masked_pixels_of_sample, outlier_check
    ) -> StatisticsContainer:
        # ensure numpy array
        masked_images = np.ma.array(images, mask=masked_pixels_of_sample)

        # mean, median, and std over the sample per pixel
        pixel_mean, pixel_median, pixel_std = sigma_clipped_stats(
            masked_images,
            sigma=self.max_sigma,
            maxiters=self.iterations,
            cenfunc="mean",
            axis=0,
        )

        # mask pixels without defined statistical values
        pixel_mean = np.ma.array(pixel_mean, mask=np.isnan(pixel_mean))
        pixel_median = np.ma.array(pixel_median, mask=np.isnan(pixel_median))
        pixel_std = np.ma.array(pixel_std, mask=np.isnan(pixel_std))

        # median of the median over the camera
        median_of_pixel_median = np.ma.median(pixel_median, axis=1)

        # median of the std over the camera
        median_of_pixel_std = np.ma.median(pixel_std, axis=1)

        # std of the std over camera
        std_of_pixel_std = np.ma.std(pixel_std, axis=1)

        # outliers from median
        image_deviation = pixel_median - median_of_pixel_median[:, np.newaxis]
        if outlier_check == "SIGNAL":
            image_median_outliers = np.logical_or(
                image_deviation
                < self.image_median_cut_outliers[0]  # pylint: disable=unsubscriptable-object
                * median_of_pixel_median[:, np.newaxis],
                image_deviation
                > self.image_median_cut_outliers[1]  # pylint: disable=unsubscriptable-object
                * median_of_pixel_median[:, np.newaxis],
            )
        elif outlier_check == "NOISE":
            image_median_outliers = np.logical_or(
                image_deviation
                < self.image_median_cut_outliers[0]  # pylint: disable=unsubscriptable-object
                * std_of_pixel_std[:, np.newaxis],
                image_deviation
                > self.image_median_cut_outliers[1]  # pylint: disable=unsubscriptable-object
                * std_of_pixel_std[:, np.newaxis],
            )

        # outliers from standard deviation
        deviation = pixel_std - median_of_pixel_std[:, np.newaxis]
        image_std_outliers = np.logical_or(
            deviation
            < self.image_std_cut_outliers[0]  # pylint: disable=unsubscriptable-object
            * std_of_pixel_std[:, np.newaxis],
            deviation
            > self.image_std_cut_outliers[1]  # pylint: disable=unsubscriptable-object
            * std_of_pixel_std[:, np.newaxis],
        )

        return StatisticsContainer(
            extraction_start=times[0],
            extraction_stop=times[-1],
            mean=pixel_mean.filled(np.nan),
            median=pixel_median.filled(np.nan),
            median_outliers=image_median_outliers.filled(True),
            std=pixel_std.filled(np.nan),
            std_outliers=image_std_outliers.filled(True),
        )
