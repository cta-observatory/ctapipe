"""
Algorithms to compute aggregated time-series statistics from columns of an astropy table.

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
    "BaseChunking",
    "SizeChunking",
    "TimeChunking",
    "BaseAggregator",
    "StatisticsAggregator",
    "PlainAggregator",
    "SigmaClippingAggregator",
]

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Generator

import astropy.units as u
import numpy as np
from astropy.stats import sigma_clip
from astropy.table import Table

from ..containers import ChunkStatisticsContainer
from ..core import Component
from ..core.traits import AstroQuantity, Bool, ComponentName, Enum, Int


class BaseChunking(Component, metaclass=ABCMeta):
    """
    Abstract base class for chunking strategies.

    Chunking components divide tables into overlapping or non-overlapping chunks
    for processing by aggregators.
    """

    allow_undersized_tables = Bool(
        default_value=False,
        help=(
            "If True, allow processing tables smaller than chunk size/duration by yielding "
            "the entire table as a single chunk. If False, raise an error."
        ),
    ).tag(config=True)

    last_chunk_policy = Enum(
        values=["overlap", "truncate", "skip"],
        default_value="overlap",
        help=(
            "Policy for handling the last chunk when data doesn't divide evenly. "
            "'overlap': Create overlapping chunk with full size (default behavior). "
            "'truncate': Yield remaining data as smaller chunk. "
            "'skip': Skip the last partial chunk."
        ),
    ).tag(config=True)

    def __call__(self, table) -> Generator[Table, None, None]:
        """
        Generate chunks from the input table.

        Parameters
        ----------
        table : astropy.table.Table
            Input table with 'time' column.

        Yields
        ------
        astropy.table.Table
            Chunks of the input table. Each chunk is a view/reference to the
            original table data, meaning modifications to chunk data will affect
            the original table.
        """
        # Basic validation that all chunking strategies need
        if "time" not in table.colnames and (
            "time_start" not in table.colnames or "time_end" not in table.colnames
        ):
            raise ValueError(
                "Table must have a 'time' column or both 'time_start' and 'time_end' columns for chunking."
            )
        yield from self._generate_chunks(table)

    @abstractmethod
    def _generate_chunks(self, table) -> Generator[Table, None, None]:
        """Generate chunks from table. Implemented by subclasses."""
        pass


class SizeChunking(BaseChunking):
    """Divides tables into chunks based on number of rows."""

    chunk_size = Int(
        default_value=None,
        allow_none=True,
        help="Number of rows per chunk. If None, use entire table as one chunk.",
    ).tag(config=True)

    chunk_shift = Int(
        default_value=None,
        allow_none=True,
        help=(
            "Number of rows to shift between consecutive chunks. "
            "If None, chunks do not overlap."
        ),
    ).tag(config=True)

    def _generate_final_chunk(self, table, last_chunk_start):
        """Generate the final chunk according to last_chunk_policy."""
        remaining_rows = len(table) - last_chunk_start

        # Only add final chunk if there are remaining rows
        if remaining_rows == 0:
            return None

        if self.last_chunk_policy == "overlap":
            # Ensure last chunk has full size by potentially overlapping
            return table[-self.chunk_size :]
        elif self.last_chunk_policy == "truncate":
            # Yield remaining rows as smaller chunk
            return table[last_chunk_start:]
        elif self.last_chunk_policy == "skip":
            # Skip the last partial chunk
            return None

        return None

    def _generate_chunks(self, table) -> Generator[Table, None, None]:
        """Generate row-count based chunks."""
        # Handle case where chunk_size is None (entire table)
        if self.chunk_size is None:
            yield table
            return

        # Check table size vs chunk_size
        if len(table) < self.chunk_size:
            if self.allow_undersized_tables:
                yield table  # Yield entire table as single chunk
                return
            else:
                raise ValueError(
                    f"Table length ({len(table)}) is less than chunk_size ({self.chunk_size}). "
                    f"Set allow_undersized_tables=True to process as single chunk."
                )

        # Calculate step size
        step = self.chunk_shift if self.chunk_shift is not None else self.chunk_size

        # Calculate all main chunk start indices (not extending beyond table)
        n_chunks = (len(table) - self.chunk_size) // step + 1
        main_chunk_indices = np.arange(n_chunks) * step

        # Filter indices that would create valid chunks
        main_chunk_indices = main_chunk_indices[
            main_chunk_indices + self.chunk_size <= len(table)
        ]
        last_chunk_start = main_chunk_indices[-1] + step

        # Generate chunks for each main chunk start index
        for start_idx in main_chunk_indices:
            end_idx = start_idx + self.chunk_size
            yield table[start_idx:end_idx]

        # Handle final chunk according to policy
        final_chunk = self._generate_final_chunk(table, last_chunk_start)
        if final_chunk is not None:
            yield final_chunk


class TimeChunking(BaseChunking):
    """Divides tables into chunks based on time intervals."""

    chunk_duration = AstroQuantity(
        physical_type=u.s,
        default_value=0 * u.s,
        help="Duration of each time chunk.",
    ).tag(config=True)

    chunk_shift = AstroQuantity(
        physical_type=u.s,
        default_value=0 * u.s,
        allow_none=True,
        help=("Time shift between consecutive chunks. If 0, chunks do not overlap."),
    ).tag(config=True)

    def _validate_inputs(self, total_duration):
        """Validate chunk_duration and table duration."""
        chunk_duration = self.chunk_duration.to_value(u.s)
        # Check if chunk_duration is properly set
        if chunk_duration <= 0:
            raise ValueError("chunk_duration must be greater than zero.")

        # Check if total duration is sufficient for chunking
        if total_duration < chunk_duration:
            if self.allow_undersized_tables:
                return True  # Signal to yield entire table as single chunk
            raise ValueError(
                f"Total duration ({total_duration * u.s}) is less than chunk_duration ({self.chunk_duration}). "
                f"Set allow_undersized_tables=True to process as single chunk."
            )
        return False  # Normal processing

    def _generate_final_chunk(self, table, last_chunk_idx):
        """Generate the final chunk according to last_chunk_policy."""
        if last_chunk_idx == len(table):
            return None
        if self.last_chunk_policy == "overlap":
            # Ensure last chunk has full duration by potentially overlapping
            last_chunk_idx = np.searchsorted(
                table["time"], table["time"][-1] - self.chunk_duration, side="left"
            )
            return table[last_chunk_idx:]
        elif self.last_chunk_policy == "truncate":
            # Yield remaining time as smaller chunk
            return table[last_chunk_idx:]
        elif self.last_chunk_policy == "skip":
            # Skip the last partial chunk
            return None

    def _generate_chunks(self, table) -> Generator[Table, None, None]:
        """Generate time-based chunks."""
        # Handle case where chunk_duration is None (entire table)
        if self.chunk_duration is None:
            yield table
            return

        times = table["time"]
        relative_times = (times - times[0]).to_value(u.s)

        # Validate inputs and handle undersized tables
        use_entire_table = self._validate_inputs(relative_times[-1])
        if use_entire_table:
            yield table  # Yield entire table as single chunk
            return

        # Calculate time step
        time_step = (
            self.chunk_shift if self.chunk_shift > 0 * u.s else self.chunk_duration
        ).to_value(u.s)

        chunk_start_time = np.arange(
            0,
            relative_times[-1] - self.chunk_duration.to_value(u.s),
            time_step,
        )
        chunk_end_time = chunk_start_time + self.chunk_duration.to_value(u.s)

        chunk_start_idx = np.searchsorted(relative_times, chunk_start_time)
        chunk_end_idx = np.searchsorted(relative_times, chunk_end_time, side="right")

        # Generate chunks for each main chunk start time
        for i, j in zip(chunk_start_idx, chunk_end_idx):
            chunk = table[i:j]
            if chunk is not None:
                yield chunk

        # Handle final chunk according to policy
        last_chunk_idx = np.searchsorted(
            relative_times, chunk_start_time[-1] + time_step
        )
        final_chunk = self._generate_final_chunk(table, last_chunk_idx)
        if final_chunk is not None:
            yield final_chunk


class BaseAggregator(Component, metaclass=ABCMeta):
    """
    Base class for aggregators that compute statistics over chunks of data.

    Aggregators use a chunking strategy to divide input tables and compute
    aggregated statistics for each chunk.
    """

    chunking_type = ComponentName(
        BaseChunking,
        default_value="SizeChunking",
        help="The chunking strategy to use for dividing data into chunks.",
    ).tag(config=True)

    def __init__(self, config=None, parent=None, **kwargs):
        """
        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments
        parent : ctapipe.core.Component or ctapipe.core.Tool
            Parent of this component in the configuration hierarchy
        """
        super().__init__(config=config, parent=parent, **kwargs)

        # Create the chunking component using ComponentName
        self.chunking = BaseChunking.from_name(self.chunking_type, parent=self)

    def __call__(
        self,
        table,
        masked_elements_of_sample=None,
        col_name="image",
    ) -> Table:
        r"""
        Divide table into chunks and compute aggregated statistic values.

        Parameters
        ----------
        table : astropy.table.Table
            table with event-wise data of shape (n_events, \*data_dimensions),
            event IDs and timestamps of shape (n_events, )
        masked_elements_of_sample : ndarray, optional
            boolean array of masked elements of shape (\*data_dimensions)
            that are not available for processing
        col_name : string
            column name in the table containing the event-wise data to aggregate

        Returns
        -------
        astropy.table.Table
            table containing the start and end values as timestamps and event IDs
            as well as the aggregated statistic values for each chunk
        """
        # Get chunks using the chunking strategy
        chunks = self.chunking(table)

        # Initialize result storage
        results = defaultdict(list)

        # Process each chunk
        for chunk in chunks:
            # Add time metadata
            if "time_start" in chunk.colnames and "time_end" in chunk.colnames:
                results["time_start"].append(chunk["time_start"][0])
                results["time_end"].append(chunk["time_end"][-1])
            else:
                results["time_start"].append(chunk["time"][0])
                results["time_end"].append(chunk["time"][-1])

            # Add event_id metadata if column exists
            if "event_id" in chunk.colnames:
                results["event_id_start"].append(chunk["event_id"][0])
                results["event_id_end"].append(chunk["event_id"][-1])

            # Compute aggregator-specific statistics
            self._add_result_columns(
                chunk[col_name].data, masked_elements_of_sample, results
            )

        # Create and return table
        result_table = Table(results)

        # Preserve units if present
        if hasattr(table[col_name], "unit") and table[col_name].unit is not None:
            self._set_result_units(result_table, table[col_name].unit)

        return result_table

    @abstractmethod
    def _add_result_columns(self, data, masked_elements_of_sample, results_dict):
        r"""
        Compute statistics and add columns to results dictionary.

        Parameters
        ----------
        data : array-like
            Data for this chunk (already extracted from table)
        masked_elements_of_sample : ndarray, optional
            Boolean mask of shape (\*data_dimensions) for elements to exclude
        results_dict : dict
            Dictionary to which statistic columns should be added.
        """
        pass

    @abstractmethod
    def _set_result_units(self, table, unit):
        """Set units for result columns that should inherit from input data."""
        pass


class StatisticsAggregator(BaseAggregator):
    """
    Base component to handle the computation of aggregated statistic values from a table
    containing any event-wise quantities (e.g., images, scalars, vectors, or other arrays).

    Aggregation is performed along axis=0 (the event dimension) for any N-dimensional data.
    """

    def _add_result_columns(self, data, masked_elements_of_sample, results_dict):
        stats = self.compute_stats(data, masked_elements_of_sample)
        results_dict["n_events"].append(stats.n_events)
        results_dict["mean"].append(stats.mean)
        results_dict["median"].append(stats.median)
        results_dict["std"].append(stats.std)

    def _set_result_units(self, table, unit):
        """
        Set units for statistics columns that inherit from the input data.

        For StatisticsAggregator, the mean, median, and std columns
        should have the same units as the input data.
        """
        for col in ("mean", "median", "std"):
            table[col].unit = unit

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
        # Mask excluded elements and NaN/inf values
        masked_data = np.ma.array(data, mask=masked_elements_of_sample)
        masked_data = np.ma.masked_invalid(masked_data)

        # Compute the mean, median, and std over the event dimension (axis=0)
        element_mean = np.ma.mean(masked_data, axis=0)
        element_median = np.ma.median(masked_data, axis=0)
        element_std = np.ma.std(masked_data, axis=0)

        # For 1D data, these operations return scalars (not MaskedArrays)
        # Convert to array and fill masked values with NaN
        element_mean = np.ma.filled(np.ma.asarray(element_mean), np.nan)
        element_median = np.ma.filled(np.ma.asarray(element_median), np.nan)
        element_std = np.ma.filled(np.ma.asarray(element_std), np.nan)

        # Count non-masked events per element (excludes both masked and NaN/inf values)
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
        # Mask excluded elements and NaN/inf values
        masked_data = np.ma.array(data, mask=masked_elements_of_sample)
        masked_data = np.ma.masked_invalid(masked_data)

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
        # (excludes both masked, NaN/inf, and sigma-clipped values)
        n_events_after_clipping = np.count_nonzero(~filtered_data.mask, axis=0)

        # Compute statistics from the filtered data along the event dimension
        element_mean = np.ma.mean(filtered_data, axis=0)
        element_median = np.ma.median(filtered_data, axis=0)
        element_std = np.ma.std(filtered_data, axis=0)

        # For 1D data, these operations return scalars (not MaskedArrays)
        # Convert to array and fill masked values with NaN
        element_mean = np.ma.filled(np.ma.asarray(element_mean), np.nan)
        element_median = np.ma.filled(np.ma.asarray(element_median), np.nan)
        element_std = np.ma.filled(np.ma.asarray(element_std), np.nan)

        return ChunkStatisticsContainer(
            n_events=n_events_after_clipping,
            mean=element_mean,
            median=element_median,
            std=element_std,
        )
