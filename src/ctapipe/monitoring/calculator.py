"""
Definition of the ``PixelStatisticsCalculator`` class, providing all steps needed to
calculate the montoring data for the camera calibration.
"""

import numpy as np
from astropy.table import Table, vstack

from ctapipe.core import TelescopeComponent
from ctapipe.core.traits import (
    ComponentName,
    Dict,
    Float,
    Int,
    List,
    TelescopeParameter,
    TraitError,
)
from ctapipe.monitoring.aggregator import StatisticsAggregator
from ctapipe.monitoring.outlier import OutlierDetector

__all__ = [
    "PixelStatisticsCalculator",
]


class PixelStatisticsCalculator(TelescopeComponent):
    """
    Component to calculate statistics from calibration events.

    The ``PixelStatisticsCalculator`` is responsible for calculating various statistics from
    calibration events, such as pedestal and flat-field data. It aggregates statistics,
    detects outliers, and handles faulty data periods.
    This class holds two functions to conduct two different passes over the data with and without
    overlapping aggregation chunks. The first pass is conducted with non-overlapping chunks,
    while overlapping chunks can be set by the ``chunk_shift`` parameter for the second pass.
    The second pass over the data is only conducted in regions of trouble with a high fraction
    of faulty pixels exceeding the threshold ``faulty_pixels_fraction``.
    """

    stats_aggregator_type = TelescopeParameter(
        trait=ComponentName(
            StatisticsAggregator, default_value="SigmaClippingAggregator"
        ),
        default_value="SigmaClippingAggregator",
        help="Name of the StatisticsAggregator subclass to be used.",
    ).tag(config=True)

    outlier_detector_list = List(
        trait=Dict(),
        default_value=None,
        allow_none=True,
        help=(
            "List of dicts containing the name of the OutlierDetector subclass to be used, "
            "the aggregated statistic value to which the detector should be applied, "
            "and the configuration of the specific detector. "
            "E.g. ``[{'apply_to': 'std', 'name': 'RangeOutlierDetector', 'config': {'validity_range': [2.0, 8.0]}}]``."
        ),
    ).tag(config=True)

    chunk_shift = Int(
        default_value=None,
        allow_none=True,
        help=(
            "Number of samples to shift the aggregation chunk for the calculation "
            "of the statistical values. Only used in the second_pass(), since the "
            "first_pass() is conducted with non-overlapping chunks (chunk_shift=None)."
        ),
    ).tag(config=True)

    faulty_pixels_fraction = Float(
        default_value=0.1,
        allow_none=True,
        help="Minimum fraction of faulty camera pixels to identify regions of trouble.",
    ).tag(config=True)

    def __init__(
        self,
        subarray,
        config=None,
        parent=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        subarray: ctapipe.instrument.SubarrayDescription
            Description of the subarray. Provides information about the
            camera which are useful in calibration. Also required for
            configuring the TelescopeParameter traitlets.
        config: traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            This is mutually exclusive with passing a ``parent``.
        parent: ctapipe.core.Component or ctapipe.core.Tool
            Parent of this component in the configuration hierarchy,
            this is mutually exclusive with passing ``config``
        """
        super().__init__(subarray=subarray, config=config, parent=parent, **kwargs)

        # Initialize the instances of StatisticsAggregator
        self.stats_aggregators = {}
        for _, _, name in self.stats_aggregator_type:
            self.stats_aggregators[name] = StatisticsAggregator.from_name(
                name, subarray=self.subarray, parent=self
            )

        # Initialize the instances of OutlierDetector from the configuration
        self.outlier_detectors, self.apply_to_list = [], []
        if self.outlier_detector_list is not None:
            for d, outlier_detector in enumerate(self.outlier_detector_list):
                # Check if all required keys are present
                missing_keys = {
                    "apply_to",
                    "name",
                    "config",
                } - outlier_detector.keys()
                if missing_keys:
                    raise TraitError(
                        f"Entry '{d}' in the ``outlier_detector_list`` trait"
                        f"is missing required key(s): {', '.join(missing_keys)}"
                    )
                self.apply_to_list.append(outlier_detector["apply_to"])
                self.outlier_detectors.append(
                    OutlierDetector.from_name(
                        outlier_detector["name"],
                        subarray=self.subarray,
                        parent=self,
                        **outlier_detector["config"],
                    )
                )

    def first_pass(
        self,
        table,
        tel_id,
        masked_pixels_of_sample=None,
        col_name="image",
    ) -> Table:
        """
        Calculate the monitoring data for a given set of events with non-overlapping aggregation chunks.

        This method performs the first pass over the provided data table to calculate
        various statistics for calibration purposes. The statistics are aggregated with
        non-overlapping chunks (``chunk_shift`` set to None), and faulty pixels are detected
        using a list of outlier detectors.


        Parameters
        ----------
        table : astropy.table.Table
            DL1-like table with images of shape (n_images, n_channels, n_pixels), event IDs and
            timestamps of shape (n_images, )
        tel_id : int
            Telescope ID for which the calibration is being performed
        masked_pixels_of_sample : ndarray, optional
            Boolean array of masked pixels of shape (n_channels, n_pixels) that are not available for processing
        col_name : str
            Column name in the table from which the statistics will be aggregated

        Returns
        -------
        astropy.table.Table
            Table containing the aggregated statistics, their outlier masks, and the validity of the chunks
        """
        # Get the aggregator
        aggregator = self.stats_aggregators[self.stats_aggregator_type.tel[tel_id]]
        # Pass through the whole provided dl1 table
        aggregated_stats = aggregator(
            table=table,
            masked_pixels_of_sample=masked_pixels_of_sample,
            col_name=col_name,
            chunk_shift=None,
        )
        # Detect faulty pixels with multiple instances of ``OutlierDetector``
        # and append the outlier masks to the aggregated statistics
        self._find_and_append_outliers(aggregated_stats)
        # Get valid chunks and add them to the aggregated statistics
        aggregated_stats["is_valid"] = self._get_valid_chunks(
            aggregated_stats["outlier_mask"]
        )
        return aggregated_stats

    def second_pass(
        self,
        table,
        valid_chunks,
        tel_id,
        masked_pixels_of_sample=None,
        col_name="image",
    ) -> Table:
        """
        Conduct a second pass over the data to refine the statistics in regions with a high percentage of faulty pixels.

        This method performs a second pass over the data with a refined shift of the chunk in regions where a high percentage
        of faulty pixels were detected during the first pass. Note: Multiple first passes of different calibration events are
        performed which may lead to different identification of faulty chunks in rare cases. Therefore a joined list of faulty
        chunks is recommended to be passed to the second pass(es) if those different passes use the same ``chunk_size``.

        Parameters
        ----------
        table : astropy.table.Table
            DL1-like table with images of shape (n_images, n_channels, n_pixels), event IDs and timestamps of shape (n_images, ).
        valid_chunks : ndarray
            Boolean array indicating the validity of each chunk from the first pass.
            Note: This boolean array can be a ``logical_and`` from multiple first passes of different calibration events.
        tel_id : int
            Telescope ID for which the calibration is being performed.
        masked_pixels_of_sample : ndarray, optional
            Boolean array of masked pixels of shape (n_channels, n_pixels) that are not available for processing.
        col_name : str
            Column name in the table from which the statistics will be aggregated.

        Returns
        -------
        astropy.table.Table
            Table containing the aggregated statistics after the second pass, their outlier masks, and the validity of the chunks.
        """
        # Check if the chunk_shift is set for the second pass
        if self.chunk_shift is None:
            raise ValueError(
                "chunk_shift must be set if second pass over the data is requested"
            )
        # Check if at least one chunk is faulty
        if np.all(valid_chunks):
            raise ValueError(
                "All chunks are valid. The second pass over the data is redundant."
            )
        # Get the aggregator
        aggregator = self.stats_aggregators[self.stats_aggregator_type.tel[tel_id]]
        # Conduct a second pass over the data
        aggregated_stats_secondpass = []
        faulty_chunks_indices = np.flatnonzero(~valid_chunks)
        for index in faulty_chunks_indices:
            # Log information of the faulty chunks
            self.log.info(
                "Faulty chunk detected in the first pass at index '%s'.", index
            )
            # Calculate the start of the slice depending on whether the previous chunk was faulty or not
            slice_start = (
                aggregator.chunk_size * index
                if index - 1 in faulty_chunks_indices
                else aggregator.chunk_size * (index - 1)
            )
            # Set the start of the slice to the first element of the dl1 table if out of bound
            # and add one ``chunk_shift``.
            slice_start = max(0, slice_start) + self.chunk_shift
            # Set the end of the slice to the last element of the dl1 table if out of bound
            # and subtract one ``chunk_shift``.
            slice_end = min(len(table) - 1, aggregator.chunk_size * (index + 2)) - (
                self.chunk_shift - 1
            )
            # Slice the dl1 table according to the previously calculated start and end.
            table_sliced = table[slice_start:slice_end]
            # Run the stats aggregator on the sliced dl1 table with a chunk_shift
            # to sample the period of trouble (carflashes etc.) as effectively as possible.
            # Checking for the length of the sliced table to be greater than the ``chunk_size``
            # since it can be smaller if the last two chunks are faulty. Note: The two last chunks
            # can be overlapping during the first pass, so we simply ignore them if there are faulty.
            if len(table_sliced) > aggregator.chunk_size:
                aggregated_stats_secondpass.append(
                    aggregator(
                        table=table_sliced,
                        masked_pixels_of_sample=masked_pixels_of_sample,
                        col_name=col_name,
                        chunk_shift=self.chunk_shift,
                    )
                )
        # Stack the aggregated statistics of each faulty chunk
        aggregated_stats_secondpass = vstack(aggregated_stats_secondpass)
        # Detect faulty pixels with multiple instances of OutlierDetector of the second pass
        # and append the outlier mask to the aggregated statistics
        self._find_and_append_outliers(aggregated_stats_secondpass)
        aggregated_stats_secondpass["is_valid"] = self._get_valid_chunks(
            aggregated_stats_secondpass["outlier_mask"]
        )
        return aggregated_stats_secondpass

    def _find_and_append_outliers(self, aggregated_stats):
        """
        Find outliers and append the masks in the aggregated statistics.

        This method detects outliers in the aggregated statistics using the
        outlier detectors defined in the configuration. Table containing the
        aggregated statistics will be appended with the outlier masks for each
        detector and a combined outlier mask.

        Parameters
        ----------
        aggregated_stats : astropy.table.Table
            Table containing the aggregated statistics.

        """
        outlier_mask = np.zeros_like(aggregated_stats["mean"], dtype=bool)
        for d, (column_name, outlier_detector) in enumerate(
            zip(self.apply_to_list, self.outlier_detectors)
        ):
            aggregated_stats[f"outlier_mask_detector_{d}"] = outlier_detector(
                aggregated_stats[column_name]
            )
            outlier_mask = np.logical_or(
                outlier_mask,
                aggregated_stats[f"outlier_mask_detector_{d}"],
            )
        aggregated_stats["outlier_mask"] = outlier_mask

    def _get_valid_chunks(self, outlier_mask):
        """
        Identify valid chunks based on the outlier mask.

        This method processes the outlier mask to determine which chunks of data
        are considered valid or faulty. A chunk is marked as faulty if the fraction
        of outlier pixels exceeds a predefined threshold ``faulty_pixels_fraction``.

        Parameters
        ----------
        outlier_mask : numpy.ndarray
            Boolean array indicating outlier pixels. The shape of the array should
            match the shape of the aggregated statistics.

        Returns
        -------
        numpy.ndarray
            Boolean array where each element indicates whether the corresponding
            chunk is valid (True) or faulty (False).
        """
        # Check if the camera has two gain channels
        if outlier_mask.shape[1] == 2:
            # Combine the outlier mask of both gain channels
            outlier_mask = np.logical_or.reduce(outlier_mask, axis=1)
        # Calculate the fraction of faulty pixels over the camera
        faulty_pixels = (
            np.count_nonzero(outlier_mask, axis=-1) / np.shape(outlier_mask)[-1]
        )
        # Check for valid chunks if the threshold is not exceeded
        valid_chunks = faulty_pixels < self.faulty_pixels_fraction
        return valid_chunks
