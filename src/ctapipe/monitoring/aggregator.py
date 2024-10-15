"""
Algorithms to compute aggregated time-series statistics from columns of an event table.

These classes take as input an events table, divide it into time chunks, which
may optionally overlap, and compute various aggregated statistics for each
chunk. The statistics include the count, mean, median, and standard deviation. The result
is a monitoring table with columns describing the start and stop time of the chunk
and the aggregated statistic values.
"""

__all__ = [
    "StatisticsAggregator",
    "PlainAggregator",
    "SigmaClippingAggregator",
]

from abc import abstractmethod
from collections import defaultdict

import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.table import Table

from ctapipe.containers import StatisticsContainer
from ctapipe.core import TelescopeComponent
from ctapipe.core.traits import Int

__all__ = [
    "StatisticsAggregator",
    "PlainAggregator",
    "SigmaClippingAggregator",
]


class StatisticsAggregator(TelescopeComponent):
    """
    Base component to handle the computation of aggregated statistic values from a table
    containing e.g. charges, peak times and/or charge variances (images).
    """

    chunk_size = Int(
        2500,
        help="Size of the chunk used for the computation of aggregated statistic values",
    ).tag(config=True)

    def __call__(
        self,
        table,
        masked_pixels_of_sample=None,
        chunk_shift=None,
        col_name="image",
    ) -> Table:
        """
        Divide table into chunks and compute aggregated statistic values.

        This function divides the input table into overlapping or non-overlapping chunks of size ``chunk_size``
        and call the relevant function of the particular aggregator to compute aggregated statistic values.
        The chunks are generated in a way that ensures they do not overflow the bounds of the table.
        - If ``chunk_shift`` is None, chunks will not overlap, but the last chunk is ensured to be
        of size ``chunk_size``, even if it means the last two chunks will overlap.
        - If ``chunk_shift`` is provided, it will determine the number of samples to shift between the start
        of consecutive chunks resulting in an overlap of chunks. Chunks that overflows the bounds
        of the table are not considered.

        Parameters
        ----------
        table : astropy.table.Table
            table with images of shape (n_images, n_channels, n_pixels), event IDs and
            timestamps of shape (n_images, )
        masked_pixels_of_sample : ndarray, optional
            boolean array of masked pixels of shape (n_channels, n_pixels) that are not available for processing
        chunk_shift : int, optional
            number of samples to shift between the start of consecutive chunks
        col_name : string
            column name in the table

        Returns
        -------
        astropy.table.Table
            table containing the start and end values as timestamps and event IDs
            as well as the aggregated statistic values (mean, median, std) for each chunk
        """

        # Check if the statistics of the table is sufficient to compute at least one complete chunk.
        if len(table) < self.chunk_size:
            raise ValueError(
                f"The length of the provided table ({len(table)}) is insufficient to meet the required statistics for a single chunk of size ({self.chunk_size})."
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

        # Compute aggregated statistic values for each chunk of images
        units = {col: table[col_name].unit for col in ("mean", "median", "std")}
        data = defaultdict(list)
        for chunk in _get_chunks(table, chunk_shift):
            stats = self.compute_stats(chunk[col_name].data, masked_pixels_of_sample)
            data["time_start"].append(chunk["time_mono"][0])
            data["time_end"].append(chunk["time_mono"][-1])
            data["event_id_start"].append(chunk["event_id"][0])
            data["event_id_end"].append(chunk["event_id"][-1])
            data["n_events"].append(stats.n_events)
            data["mean"].append(stats.mean)
            data["median"].append(stats.median)
            data["std"].append(stats.std)

        return Table(data, units=units)

    @abstractmethod
    def compute_stats(self, images, masked_pixels_of_sample) -> StatisticsContainer:
        pass


class PlainAggregator(StatisticsAggregator):
    """
    Compute aggregated statistic values from a chunk of images using numpy functions
    """

    def compute_stats(self, images, masked_pixels_of_sample) -> StatisticsContainer:
        # Mask broken pixels
        masked_images = np.ma.array(images, mask=masked_pixels_of_sample)

        # Compute the mean, median, and std over the chunk per pixel
        pixel_mean = np.ma.mean(masked_images, axis=0).filled(np.nan)
        pixel_median = np.ma.median(masked_images, axis=0).filled(np.nan)
        pixel_std = np.ma.std(masked_images, axis=0).filled(np.nan)

        return StatisticsContainer(
            n_events=masked_images.shape[0],
            mean=pixel_mean,
            median=pixel_median,
            std=pixel_std,
        )


class SigmaClippingAggregator(StatisticsAggregator):
    """
    Compute aggregated statistic values from a chunk of images using astropy's sigma clipping functions
    """

    max_sigma = Int(
        default_value=4,
        help="Maximal value for the sigma clipping outlier removal",
    ).tag(config=True)
    iterations = Int(
        default_value=5,
        help="Number of iterations for the sigma clipping outlier removal",
    ).tag(config=True)

    def compute_stats(self, images, masked_pixels_of_sample) -> StatisticsContainer:
        # Mask broken pixels
        masked_images = np.ma.array(images, mask=masked_pixels_of_sample)

        # Compute the mean, median, and std over the chunk per pixel
        pixel_mean, pixel_median, pixel_std = sigma_clipped_stats(
            masked_images,
            sigma=self.max_sigma,
            maxiters=self.iterations,
            cenfunc="mean",
            axis=0,
        )

        return StatisticsContainer(
            n_events=masked_images.shape[0],
            mean=pixel_mean,
            median=pixel_median,
            std=pixel_std,
        )
