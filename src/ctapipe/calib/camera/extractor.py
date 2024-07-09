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
    CaselessStrEnum,
    Int,
    List,
)


class StatisticsExtractor(TelescopeComponent):
    """
    Base component to handle the extraction of the statistics from a dl1 table
    containing charges, peak times and/or charge variances (images).
    """

    chunk_size = Int(
        2500,
        help="Size of the chunk used for the calculation of the statistical values",
    ).tag(config=True)

    def __call__(
        self,
        dl1_table,
        masked_pixels_of_sample=None,
        chunk_shift=None,
        col_name="image",
    ) -> list:
        """
        Prepare the extraction chunks and call the relevant function of the particular extractor
        to extract the statistical values.

        Parameters
        ----------
        dl1_table : astropy.table.Table
            DL1 table with images of shape (n_images, n_channels, n_pix)
            and timestamps of shape (n_images, ) stored in an astropy Table
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

        # Check if the length of the dl1 table is greater or equal than the size of the chunk.
        if len(dl1_table[col_name]) < self.chunk_size:
            raise ValueError(
                f"The length of the DL1 table ({len(dl1_table[col_name])}) must be greater or equal than the size of the chunk ({self.chunk_size})."
            )

        # Function to split the dl1 table into appropriated chunks
        def _get_chunks(dl1_table, chunk_shift):
            if chunk_shift is None:
                return (
                    (
                        dl1_table[i : i + self.chunk_size]
                        if i + self.chunk_size <= len(dl1_table)
                        else dl1_table[
                            len(dl1_table) - self.chunk_size : len(dl1_table)
                        ]
                    )
                    for i in range(0, len(dl1_table), self.chunk_size)
                )
            else:
                return (
                    dl1_table[i : i + self.chunk_size]
                    for i in range(0, len(dl1_table) - self.chunk_size, chunk_shift)
                )

        # Get the chunks of the dl1 table
        dl1_chunks = _get_chunks(dl1_table, chunk_shift)

        # Calculate the statistics from a chunk of images
        stats_list = [
            self.extract(
                chunk[col_name].data, chunk["time_mono"], masked_pixels_of_sample
            )
            for chunk in dl1_chunks
        ]

        return stats_list

    @abstractmethod
    def extract(self, images, times, masked_pixels_of_sample) -> StatisticsContainer:
        pass


class PlainExtractor(StatisticsExtractor):
    """
    Extract the statistics from a chunk of peak time images
    using numpy and scipy functions
    """

    time_median_cut_outliers = List(
        [0, 60],
        help="Interval (in waveform samples) of accepted median peak time values",
    ).tag(config=True)

    def extract(self, images, times, masked_pixels_of_sample) -> StatisticsContainer:
        # ensure numpy array
        masked_images = np.ma.array(images, mask=masked_pixels_of_sample)

        # median over the chunk per pixel
        pixel_median = np.ma.median(masked_images, axis=0)

        # mean over the chunk per pixel
        pixel_mean = np.ma.mean(masked_images, axis=0)

        # std over the chunk per pixel
        pixel_std = np.ma.std(masked_images, axis=0)

        # outliers from median
        median_outliers = np.logical_or(
            pixel_median < self.time_median_cut_outliers[0],
            pixel_median > self.time_median_cut_outliers[1],
        )

        return StatisticsContainer(
            extraction_start=times[0],
            extraction_stop=times[-1],
            mean=pixel_mean.filled(np.nan),
            median=pixel_median.filled(np.nan),
            median_outliers=median_outliers.filled(True),
            std=pixel_std.filled(np.nan),
            std_outliers=np.full(np.shape(median_outliers), False),
        )


class SigmaClippingExtractor(StatisticsExtractor):
    """
    Extract the statistics from a chunk of charge or variance images
    using astropy's sigma clipping functions
    """

    median_outliers_interval = List(
        [-0.3, 0.3],
        help=(
            "Interval of the multiplicative factor for detecting outliers based on"
            "the deviation of the median distribution."
            "- If `outlier_method` is `median`, the factors are multiplied by"
            "  the camera median of pixel medians to set the thresholds for identifying outliers."
            "- If `outlier_method` is `standard_deviation`, the factors are multiplied by"
            "  the camera standard deviation of pixel medians to set the thresholds for identifying outliers."
        ),
    ).tag(config=True)
    outlier_method = CaselessStrEnum(
        values=["median", "standard_deviation"],
        help="Method used for detecting outliers based on the deviation of the median distribution",
    ).tag(config=True)
    std_outliers_interval = List(
        [-3, 3],
        help=(
            "Interval of the multiplicative factor for detecting outliers based on"
            "the deviation of the standard deviation distribution."
            "The factors are multiplied by the camera standard deviation of pixel standard deviations"
            "to set the thresholds for identifying outliers."
        ),
    ).tag(config=True)
    max_sigma = Int(
        default_value=4,
        help="Maximal value for the sigma clipping outlier removal",
    ).tag(config=True)
    iterations = Int(
        default_value=5,
        help="Number of iterations for the sigma clipping outlier removal",
    ).tag(config=True)

    def extract(self, images, times, masked_pixels_of_sample) -> StatisticsContainer:
        # Mask broken pixels
        masked_images = np.ma.array(images, mask=masked_pixels_of_sample)

        # Calculate the mean, median, and std over the chunk per pixel
        pix_mean, pix_median, pix_std = sigma_clipped_stats(
            masked_images,
            sigma=self.max_sigma,
            maxiters=self.iterations,
            cenfunc="mean",
            axis=0,
        )

        # Mask pixels without defined statistical values
        pix_mean = np.ma.array(pix_mean, mask=np.isnan(pix_mean))
        pix_median = np.ma.array(pix_median, mask=np.isnan(pix_median))
        pix_std = np.ma.array(pix_std, mask=np.isnan(pix_std))

        # Camera median of the pixel medians
        cam_median_of_pix_median = np.ma.median(pix_median, axis=1)

        # Camera median of the pixel stds
        cam_median_of_pix_std = np.ma.median(pix_std, axis=1)

        # Camera std of the pixel stds
        cam_std_of_pix_std = np.ma.std(pix_std, axis=1)

        # Detect outliers based on the deviation of the median distribution
        median_deviation = pix_median - cam_median_of_pix_median[:, np.newaxis]
        if self.outlier_method == "median":
            median_outliers = np.logical_or(
                median_deviation
                < self.median_outliers_interval[0]
                * cam_median_of_pix_median[:, np.newaxis],
                median_deviation
                > self.median_outliers_interval[1]
                * cam_median_of_pix_median[:, np.newaxis],
            )
        elif self.outlier_method == "standard_deviation":
            # Camera std of pixel medians
            cam_std_of_pix_median = np.ma.std(pix_median, axis=1)
            median_outliers = np.logical_or(
                median_deviation
                < self.median_outliers_interval[0]
                * cam_std_of_pix_median[:, np.newaxis],
                median_deviation
                > self.median_outliers_interval[1]
                * cam_std_of_pix_median[:, np.newaxis],
            )

        # Detect outliers based on the deviation of the standard deviation distribution
        std_deviation = pix_std - cam_median_of_pix_std[:, np.newaxis]
        std_outliers = np.logical_or(
            std_deviation
            < self.std_outliers_interval[0] * cam_std_of_pix_std[:, np.newaxis],
            std_deviation
            > self.std_outliers_interval[1] * cam_std_of_pix_std[:, np.newaxis],
        )

        return StatisticsContainer(
            extraction_start=times[0],
            extraction_stop=times[-1],
            mean=pix_mean.filled(np.nan),
            median=pix_median.filled(np.nan),
            median_outliers=median_outliers.filled(True),
            std=pix_std.filled(np.nan),
            std_outliers=std_outliers.filled(True),
        )
