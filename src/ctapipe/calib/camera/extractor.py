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
from ctapipe.core.traits import Int


class StatisticsExtractor(TelescopeComponent):
    """
    Base component to handle the extraction of the statistics from a table
    containing e.g. charges, peak times and/or charge variances (images).
    """

    chunk_size = Int(
        2500,
        help="Size of the chunk used for the calculation of the statistical values",
    ).tag(config=True)

    def __call__(
        self,
        table,
        masked_pixels_of_sample=None,
        chunk_shift=None,
        col_name="image",
    ) -> list:
        """
        Divide table into chunks and extract the statistical values.

        This function divides the input table into overlapping or non-overlapping chunks of size `chunk_size`
        and call the relevant function of the particular extractor to extract the statistical values.
        The chunks are generated in a way that ensures they do not overflow the bounds of the table.
        - If `chunk_shift` is None, extraction chunks will not overlap, but the last chunk is ensured to be
        of size `chunk_size`, even if it means the last two chunks will overlap.
        - If `chunk_shift` is provided, it will determine the number of samples to shift between the start
        of consecutive chunks resulting in an overlap of extraction chunks. Chunks that overflows the bounds
        of the table are not considered.

        Parameters
        ----------
        table : astropy.table.Table
            table with images of shape (n_images, n_channels, n_pix)
            and timestamps of shape (n_images, ) stored in an astropy Table
        masked_pixels_of_sample : ndarray, optional
            boolean array of masked pixels of shape (n_pix, ) that are not available for processing
        chunk_shift : int, optional
            number of samples to shift between the start of consecutive extraction chunks
        col_name : string
            column name in the table

        Returns
        -------
        List StatisticsContainer:
            List of extracted statistics and extraction chunks
        """

        # Check if the statistics of the table is sufficient to extract at least one chunk.
        if len(table) < self.chunk_size:
            raise ValueError(
                f"The length of the provided table ({len(table)}) is insufficient to meet the required statistics for a single extraction chunk of size ({self.chunk_size})."
            )
        # Check if the chunk_shift is smaller than the chunk_size
        if chunk_shift is not None and chunk_shift > self.chunk_size:
            raise ValueError(
                f"The chunk_shift ({chunk_shift}) must be smaller than the chunk_size ({self.chunk_size})."
            )

        # Function to split the table into appropriated chunks
        def _get_chunks(table, chunk_shift):
            # Calculate the range step: Use chunk_shift if provided, otherwise use chunk_size
            step = chunk_shift or self.chunk_size

            # Generate chunks that do not overflow
            for i in range(0, len(table) - self.chunk_size + 1, step):
                yield table[i : i + self.chunk_size]

            # If chunk_shift is None, ensure the last chunk is of size chunk_size, if needed
            if chunk_shift is None and len(table) % self.chunk_size != 0:
                yield table[-self.chunk_size :]

        # Get the chunks of the table
        chunks = _get_chunks(table, chunk_shift)

        # Calculate the statistics from a chunk of images
        chunk_stats = {
            chunk["time_mono"][0]: self.extract(
                chunk[col_name].data, masked_pixels_of_sample
            )
            for chunk in chunks
        }

        return chunk_stats

    @abstractmethod
    def extract(self, images, masked_pixels_of_sample) -> StatisticsContainer:
        pass


class PlainExtractor(StatisticsExtractor):
    """
    Extract the statistics from a chunk of images using numpy functions
    """

    def extract(self, images, masked_pixels_of_sample) -> StatisticsContainer:
        # Mask broken pixels
        masked_images = np.ma.array(images, mask=masked_pixels_of_sample)

        # Calculate the mean, median, and std over the chunk per pixel
        pixel_mean = np.ma.mean(masked_images, axis=0)
        pixel_median = np.ma.median(masked_images, axis=0)
        pixel_std = np.ma.std(masked_images, axis=0)

        return StatisticsContainer(
            mean=pixel_mean.filled(np.nan),
            median=pixel_median.filled(np.nan),
            std=pixel_std.filled(np.nan),
        )


class SigmaClippingExtractor(StatisticsExtractor):
    """
    Extract the statistics from a chunk of images using astropy's sigma clipping functions
    """

    max_sigma = Int(
        default_value=4,
        help="Maximal value for the sigma clipping outlier removal",
    ).tag(config=True)
    iterations = Int(
        default_value=5,
        help="Number of iterations for the sigma clipping outlier removal",
    ).tag(config=True)

    def extract(self, images, masked_pixels_of_sample) -> StatisticsContainer:
        # Mask broken pixels
        masked_images = np.ma.array(images, mask=masked_pixels_of_sample)

        # Calculate the mean, median, and std over the chunk per pixel
        pixel_mean, pixel_median, pixel_std = sigma_clipped_stats(
            masked_images,
            sigma=self.max_sigma,
            maxiters=self.iterations,
            cenfunc="mean",
            axis=0,
        )

        return StatisticsContainer(
            mean=pixel_mean,
            median=pixel_median,
            std=pixel_std,
        )
