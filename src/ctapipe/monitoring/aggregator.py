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
    "BaseChunking",
    "SizeChunking",
    "TimeChunking",
    "BaseAggregator",
    "StatisticsAggregator",
    "PlainAggregator",
    "SigmaClippingAggregator",
]

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Generator

import astropy.units as u
import numpy as np
from astropy.stats import sigma_clip
from astropy.table import Table

from ..containers import ChunkStatisticsContainer
from ..core import Component
from ..core.traits import AstroQuantity, ComponentName, Int


class BaseChunking(Component, ABC):
    """
    Abstract base class for chunking strategies.

    Chunking components divide tables into overlapping or non-overlapping chunks
    for processing by aggregators.
    """

    def __call__(self, table) -> Generator[Table, None, None]:
        """
        Generate chunks from the input table.

        Parameters
        ----------
        table : astropy.table.Table
            Input table with 'time' and 'event_id' columns

        Yields
        ------
        astropy.table.Table
            Chunks of the input table
        """
        self._validate_table(table)
        yield from self._generate_chunks(table)

    def _validate_table(self, table):
        """Validate that table has required columns."""
        required_cols = ["time", "event_id"]
        missing = [col for col in required_cols if col not in table.colnames]
        if missing:
            raise ValueError(f"Table must have columns: {missing}")

    @abstractmethod
    def _generate_chunks(self, table) -> Generator[Table, None, None]:
        """Generate chunks from table. Implemented by subclasses."""
        pass


class SizeChunking(BaseChunking):
    """Divides tables into chunks based on number of events."""

    chunk_size = Int(
        default_value=None,
        allow_none=True,
        help="Number of events per chunk. If None, use entire table as one chunk.",
    ).tag(config=True)

    chunk_shift = Int(
        default_value=None,
        allow_none=True,
        help=(
            "Number of events to shift between consecutive chunks. "
            "If None, chunks do not overlap."
        ),
    ).tag(config=True)

    def _generate_chunks(self, table) -> Generator[Table, None, None]:
        """Generate event-count based chunks."""
        # Handle case where chunk_size is None (entire table)
        if self.chunk_size is None:
            yield table
            return

        # Validate chunk_size vs table length
        if len(table) < self.chunk_size:
            raise ValueError(
                f"Table length ({len(table)}) is less than chunk_size ({self.chunk_size})"
            )

        # Calculate step size
        step = self.chunk_shift if self.chunk_shift is not None else self.chunk_size

        # Generate overlapping/non-overlapping chunks
        for i in range(0, len(table) - self.chunk_size + 1, step):
            yield table[i : i + self.chunk_size]

        # Handle last chunk for non-overlapping case
        if self.chunk_shift is None and len(table) % self.chunk_size != 0:
            # Ensure last chunk has full size by potentially overlapping
            yield table[-self.chunk_size :]


class TimeChunking(BaseChunking):
    """Divides tables into chunks based on time intervals."""

    chunk_duration = AstroQuantity(
        physical_type=u.s,
        default_value=0 * u.s,
        help="Duration of each time chunk. If None, use entire table as one chunk.",
    ).tag(config=True)

    chunk_shift = AstroQuantity(
        physical_type=u.s,
        default_value=0 * u.s,
        allow_none=True,
        help=("Time shift between consecutive chunks. If None, chunks do not overlap."),
    ).tag(config=True)

    time_tolerance = AstroQuantity(
        physical_type=u.s,
        default_value=0.1 * u.s,
        help=(
            "Time tolerance for floating point comparisons when determining "
            "chunk boundaries. Used bidirectionally: prevents generating chunks "
            "when within tolerance of the end time, and prevents generating "
            "additional overlapping chunks when remaining time is within tolerance."
        ),
    ).tag(config=True)

    def _generate_chunks(self, table) -> Generator[Table, None, None]:
        """Generate time-based chunks."""
        # Handle case where chunk_duration is None (entire table)
        if self.chunk_duration is None:
            yield table
            return

        times = table["time"]
        start_time = times[0]
        end_time = times[-1]
        total_duration = end_time - start_time

        # Validate inputs
        if total_duration < self.chunk_duration:
            raise ValueError(
                f"Total duration ({total_duration}) is less than chunk_duration ({self.chunk_duration})"
            )
        if self.chunk_duration == 0 * u.s:
            raise ValueError("chunk_duration must be greater than zero.")

        # Calculate time step
        time_step = (
            self.chunk_shift if self.chunk_shift > 0 * u.s else self.chunk_duration
        )

        # Generate main sequence of chunks
        current_time = start_time
        while True:
            chunk_end = current_time + self.chunk_duration

            # Check if we've reached the end
            if chunk_end > end_time + self.time_tolerance:
                break

            # Create mask for this time window
            mask = (times >= current_time) & (times < chunk_end)
            if np.any(mask):
                yield table[mask]

            current_time += time_step

        # Handle last chunk for non-overlapping case only
        if self.chunk_shift == 0 * u.s:
            remaining_duration = end_time - current_time

            # Only generate last chunk if remaining duration is beyond tolerance
            # This ensures bidirectional tolerance: no chunk if we're close to the end
            if (
                remaining_duration > self.time_tolerance
                and total_duration > self.chunk_duration
            ):
                # Ensure last chunk has full duration by potentially overlapping
                last_chunk_start = end_time - self.chunk_duration
                mask = (times >= last_chunk_start) & (times <= end_time)
                if np.any(mask):
                    yield table[mask]


class BaseAggregator(Component, ABC):
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
            # Add time/event metadata
            results["time_start"].append(chunk["time"][0])
            results["time_end"].append(chunk["time"][-1])
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

    This class provides backward compatibility by wrapping add_result_columns() to call
    the existing compute_stats() method.
    """

    def _add_result_columns(self, data, masked_elements_of_sample, results_dict):
        """
        Compute statistics using compute_stats and add columns to results dictionary.

        This method provides the bridge between the new BaseAggregator interface
        and the existing compute_stats() method used by subclasses.
        """
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
            if col in table.colnames:
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
