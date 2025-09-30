"""
Algorithms to compute aggregated time-series statistics from columns of an event table.

These classes take as input an events table containing any event-wise quantities
(e.g., images, scalars, vectors), divide it into time chunks, which may optionally
overlap, and compute various aggregated statistics for each chunk. The statistics
include the count, mean, median, and standard deviation. The result is a monitoring
table with columns describing the start and stop time of the chunk and the
aggregated statistic values.

The aggregation is always performed along axis=0 (the event dimension), making
these classes suitable for any N-dimensional event-wise data.
"""

__all__ = [
    "StatisticsAggregator",
    "PlainAggregator",
    "SigmaClippingAggregator",
]

from abc import abstractmethod
from collections import defaultdict

import astropy.units as u
import numpy as np
from astropy.stats import sigma_clip
from astropy.table import Table

from ..containers import ChunkStatisticsContainer
from ..core import Component
from ..core.traits import CaselessStrEnum, Int

__all__ = [
    "StatisticsAggregator",
    "PlainAggregator",
    "SigmaClippingAggregator",
]


class StatisticsAggregator(Component):
    """
    Base component to handle the computation of aggregated statistic values from a table
    containing any event-wise quantities (e.g., images, scalars, vectors, or other arrays).

    Aggregation is performed along axis=0 (the event dimension) for any N-dimensional data.
    """

    chunk_size = Int(
        default_value=None,
        allow_none=True,
        help=(
            "Size of the chunk used for the computation of aggregated statistic values. "
            "If None, use the entire table as one chunk. "
            "For event-based chunking: number of events per chunk. "
            "For time-based chunking: duration in seconds per chunk (integer)."
        ),
    ).tag(config=True)

    chunking_mode = CaselessStrEnum(
        ["events", "time"],
        default_value="events",
        help=(
            "Chunking strategy: 'events' for fixed event count, 'time' for time intervals. "
            "When 'time', chunk_size and chunk_shift are interpreted as seconds."
        ),
    ).tag(config=True)

    def __call__(
        self,
        table,
        masked_elements_of_sample=None,
        chunk_shift=None,
        col_name="image",
    ) -> Table:
        r"""
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
            table with event-wise data of shape (n_events, \*data_dimensions), event IDs and
            timestamps of shape (n_events, )
        masked_elements_of_sample : ndarray, optional
            boolean array of masked elements of shape (\*data_dimensions) that are not available for processing
        chunk_shift : int, optional
            number of samples to shift between the start of consecutive chunks.
            For event-based chunking: number of events. For time-based chunking: seconds.
            Ignored when chunk_size is None.
        col_name : string
            column name in the table containing the event-wise data to aggregate

        Returns
        -------
        astropy.table.Table
            table containing the start and end values as timestamps and event IDs
            as well as the aggregated statistic values (mean, median, std) for each chunk
        """
        # Set chunk_size to table length if None (use entire table as one chunk)
        if self.chunk_size is None:
            effective_chunk_size = len(table)
            self.chunking_mode = (
                "events"  # Force event-based chunking when using entire table
            )
        else:
            effective_chunk_size = self.chunk_size
            self._check_table_length(table, effective_chunk_size)
        if chunk_shift is not None and chunk_shift > effective_chunk_size:
            raise ValueError(
                f"The chunk_shift ({chunk_shift}) must be smaller than the chunk_size ({effective_chunk_size})."
            )

        # Function to split the table into appropriated chunks
        def _get_chunks(table, chunk_shift, effective_chunk_size):
            # If using entire table as one chunk, just yield the whole table
            if self.chunk_size is None:
                yield table
                return

            if self.chunking_mode == "events":
                yield from self._get_event_chunks(
                    table, chunk_shift, effective_chunk_size
                )
            elif self.chunking_mode == "time":
                yield from self._get_time_chunks(
                    table, chunk_shift, effective_chunk_size
                )

        # Compute aggregated statistic values for each chunk of data
        data = defaultdict(list)
        for chunk in _get_chunks(table, chunk_shift, effective_chunk_size):
            stats = self.compute_stats(chunk[col_name].data, masked_elements_of_sample)
            data["time_start"].append(chunk["time_mono"][0])
            data["time_end"].append(chunk["time_mono"][-1])
            data["event_id_start"].append(chunk["event_id"][0])
            data["event_id_end"].append(chunk["event_id"][-1])
            data["n_events"].append(stats.n_events)
            data["mean"].append(stats.mean)
            data["median"].append(stats.median)
            data["std"].append(stats.std)

        # Create table and set units for statistical columns
        result_table = Table(data)
        if hasattr(table[col_name], "unit") and table[col_name].unit is not None:
            for col in ("mean", "median", "std"):
                result_table[col].unit = table[col_name].unit

        return result_table

    def _check_table_length(self, table, effective_chunk_size):
        """Check if the table length is sufficient for at least one chunk."""
        if self.chunking_mode == "events":
            if len(table) < effective_chunk_size:
                raise ValueError(
                    f"The length of the provided table ({len(table)}) "
                    f"is insufficient to meet the required statistics "
                    f"for a single chunk of size ({effective_chunk_size})."
                )
        elif self.chunking_mode == "time":
            times = table["time_mono"]
            total_duration = (times[-1] - times[0]).to_value("s")
            if total_duration < effective_chunk_size:
                raise ValueError(
                    f"The total duration of the provided table"
                    f"({total_duration} seconds) is insufficient "
                    f"to meet the required statistics for a single chunk "
                    f"of size ({effective_chunk_size} seconds)."
                )

    def _get_event_chunks(self, table, chunk_shift, effective_chunk_size):
        """Generate chunks based on event count."""
        # Calculate the range step: Use chunk_shift if provided, otherwise use chunk_size
        step = chunk_shift or effective_chunk_size

        # Generate chunks that do not overflow
        for i in range(0, len(table) - effective_chunk_size + 1, step):
            yield table[i : i + effective_chunk_size]

        # If chunk_shift is None, ensure the last chunk is of size chunk_size, if needed
        if chunk_shift is None and len(table) % effective_chunk_size != 0:
            yield table[-effective_chunk_size:]

    def _get_time_chunks(self, table, chunk_shift, effective_chunk_size):
        """Generate chunks based on time intervals."""
        times = table["time_mono"]
        start_time = times[0]
        end_time = times[-1]
        total_duration = (end_time - start_time).to_value("s")

        # Calculate time step: Use chunk_shift if provided, otherwise use chunk_size
        time_step = chunk_shift if chunk_shift is not None else effective_chunk_size

        current_time = start_time
        while True:
            chunk_end = current_time + effective_chunk_size * u.s
            # Check if chunk end exceeds our data range (with 0.1s tolerance for floating point)
            if chunk_end > end_time + 0.1 * u.s:
                break

            mask = (times >= current_time) & (times < chunk_end)
            if np.any(mask):
                yield table[mask]
            current_time += time_step * u.s

        if chunk_shift is None:
            # Add overlapping last chunk if there's remaining data
            remaining_duration = (end_time - current_time).to_value("s")
            if remaining_duration > 0.1 and total_duration > effective_chunk_size:
                # Ensure last chunk has full duration by potentially overlapping
                last_chunk_start = end_time - effective_chunk_size * u.s
                mask = (times >= last_chunk_start) & (times <= end_time)
                if np.any(mask):
                    yield table[mask]

    @abstractmethod
    def compute_stats(
        self, data, masked_elements_of_sample
    ) -> ChunkStatisticsContainer:
        r"""
        Compute aggregated statistics for a chunk of data.

        Parameters
        ----------
        data : ndarray
            Event-wise data of shape (n_events, \*data_dimensions)
        masked_elements_of_sample : ndarray, optional
            Boolean mask of shape (\*data_dimensions) for elements to exclude

        Returns
        -------
        StatisticsContainer
            Container with computed statistics
        """
        pass


class PlainAggregator(StatisticsAggregator):
    """
    Compute aggregated statistic values from a chunk of event-wise data using numpy functions.

    Works with any N-dimensional event-wise data by aggregating along axis=0 (event dimension).
    """

    def compute_stats(
        self, data, masked_elements_of_sample
    ) -> ChunkStatisticsContainer:
        # Mask excluded elements
        masked_data = np.ma.array(data, mask=masked_elements_of_sample)

        # Compute the mean, median, and std over the event dimension (axis=0)
        element_mean = np.ma.mean(masked_data, axis=0).filled(np.nan)
        element_median = np.ma.median(masked_data, axis=0).filled(np.nan)
        element_std = np.ma.std(masked_data, axis=0).filled(np.nan)

        # Count non-masked events per element (for consistency with SigmaClippingAggregator)
        n_events_per_element = np.count_nonzero(~masked_data.mask, axis=0)

        return ChunkStatisticsContainer(
            n_events=n_events_per_element,
            mean=element_mean,
            median=element_median,
            std=element_std,
        )


class SigmaClippingAggregator(StatisticsAggregator):
    """
    Compute aggregated statistic values from a chunk of event-wise data using astropy's sigma clipping functions.

    Works with any N-dimensional event-wise data by aggregating along axis=0 (event dimension)
    while removing outliers using sigma clipping.
    """

    max_sigma = Int(
        default_value=4,
        help="Maximal value for the sigma clipping outlier removal",
    ).tag(config=True)
    iterations = Int(
        default_value=5,
        help="Number of iterations for the sigma clipping outlier removal",
    ).tag(config=True)

    def compute_stats(
        self, data, masked_elements_of_sample
    ) -> ChunkStatisticsContainer:
        # Mask excluded elements
        masked_data = np.ma.array(data, mask=masked_elements_of_sample)

        # Use sigma_clip to get the clipped data, then compute stats from it
        # Clipping is performed along axis=0 (event dimension)
        filtered_data = sigma_clip(
            masked_data,
            sigma=self.max_sigma,
            maxiters=self.iterations,
            cenfunc="mean",
            axis=0,
        )

        # Count the number of events remaining after sigma clipping per element
        n_events_after_clipping = np.count_nonzero(~filtered_data.mask, axis=0)

        # Compute statistics from the filtered data along the event dimension
        element_mean = np.ma.mean(filtered_data, axis=0).filled(np.nan)
        element_median = np.ma.median(filtered_data, axis=0).filled(np.nan)
        element_std = np.ma.std(filtered_data, axis=0).filled(np.nan)

        return ChunkStatisticsContainer(
            n_events=n_events_after_clipping,
            mean=element_mean,
            median=element_median,
            std=element_std,
        )
