"""
Definition of the ``PixelStatisticsCalculator`` class, providing all steps needed to
calculate the monitoring data for the camera calibration.
"""

import numpy as np
from astropy.table import Table, vstack
from astropy.time import Time

from ..core import TelescopeComponent
from ..core.traits import (
    ComponentName,
    Dict,
    Float,
    List,
    TelescopeParameter,
    TraitError,
)
from .aggregator import BaseAggregator
from .outlier import OutlierDetector

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
        trait=ComponentName(BaseAggregator, default_value="SigmaClippingAggregator"),
        default_value="SigmaClippingAggregator",
        help="Name of the BaseAggregator subclass to be used.",
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

        # Initialize the instances of BaseAggregator
        self.stats_aggregators = {}
        for _, _, name in self.stats_aggregator_type:
            self.stats_aggregators[name] = BaseAggregator.from_name(name, parent=self)

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

    def __call__(
        self, table, tel_id, masked_pixels_of_sample=None, col_name="image"
    ) -> Table:
        """Calculate statistics and refine faulty periods if overlap is configured.

        Parameters are the same as for :meth:`first_pass`.
        """
        stats = self.first_pass(
            table=table,
            tel_id=tel_id,
            masked_pixels_of_sample=masked_pixels_of_sample,
            col_name=col_name,
        )
        aggregator = self.stats_aggregators[self.stats_aggregator_type.tel[tel_id]]
        if not aggregator.chunking.has_overlap:
            return stats
        if np.all(stats["is_valid"]):
            self.log.info(
                "No faulty chunks found for telescope 'tel_id=%d'. Skipping second pass.",
                tel_id,
            )
            return stats

        refined = self.second_pass(
            table=table,
            valid_chunks=stats["is_valid"].data,
            tel_id=tel_id,
            masked_elements_of_sample=masked_pixels_of_sample,
            col_name=col_name,
        )
        combined = vstack([stats, refined])
        combined.sort("time_start")
        return combined

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
        unshifted chunks, and faulty pixels are detected
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

        aggregated_stats = aggregator(
            table=table,
            chunks=aggregator.chunking(table, overlap=False),
            masked_elements_of_sample=masked_pixels_of_sample,
            col_name=col_name,
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
        masked_elements_of_sample=None,
        col_name="image",
    ) -> Table:
        """
        Conduct a second pass over the data to refine the statistics or histograms in regions with a high percentage of faulty pixels.

        This method performs a second pass over the data with a refined shift of the chunk in regions where a high percentage
        of faulty pixels were detected during the first pass. Windows are generated on the configured overlapping grid
        for the full input table, retaining only windows intersecting faulty first-pass chunks and excluding windows
        already calculated in the first pass. Note: Multiple first passes of different calibration events are
        performed which may lead to different identification of faulty chunks in rare cases. Therefore a joined list of faulty
        chunks may be passed to the second passes if they use identical first-pass chunk boundaries.

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
            Table containing the aggregated statistics or histograms after the second pass, their outlier masks, and the validity of the chunks.
        """
        # Check if at least one chunk is faulty
        if np.all(valid_chunks):
            raise ValueError(
                "All chunks are valid. The second pass over the data is redundant."
            )
        # Get the aggregator
        aggregator = self.stats_aggregators[self.stats_aggregator_type.tel[tel_id]]

        if not aggregator.chunking.has_overlap:
            raise ValueError(
                "Second pass requires overlapping chunks to be configured."
            )

        # Recreate actual first-pass boundaries without changing configuration.
        first_pass_chunks = list(aggregator.chunking(table, overlap=False))

        valid_chunks = np.asarray(valid_chunks, dtype=bool)
        if valid_chunks.shape != (len(first_pass_chunks),):
            raise ValueError("valid_chunks must match the number of first-pass chunks.")

        first_starts = Time([chunk["time"][0] for chunk in first_pass_chunks])
        first_ends = Time([chunk["time"][-1] for chunk in first_pass_chunks])
        faulty_starts = first_starts[~valid_chunks]
        faulty_ends = first_ends[~valid_chunks]

        def additional_chunks():
            seen = set()
            for chunk in aggregator.chunking(table):
                start, end = chunk["time"][[0, -1]]
                # Keep only new windows containing data from a faulty period.
                if not np.any((start <= faulty_ends) & (end >= faulty_starts)):
                    continue
                if np.any((start == first_starts) & (end == first_ends)):
                    continue
                key = (start.jd1, start.jd2, end.jd1, end.jd2)
                if key not in seen:
                    seen.add(key)
                    yield chunk

        chunks = list(additional_chunks())
        # Use one real chunk to preserve the output schema and units even when
        # there are no additional windows (e.g. an undersized single chunk).
        aggregated_stats_secondpass = aggregator(
            table=table,
            chunks=chunks or first_pass_chunks[:1],
            masked_elements_of_sample=masked_elements_of_sample,
            col_name=col_name,
        )
        if not chunks:
            aggregated_stats_secondpass = aggregated_stats_secondpass[:0]
        # Detect faulty pixels with multiple instances of OutlierDetector of the second pass
        # and append the outlier mask to the aggregated statistics
        self._find_and_append_outliers(aggregated_stats_secondpass)
        aggregated_stats_secondpass["is_valid"] = self._get_valid_chunks(
            aggregated_stats_secondpass["outlier_mask"]
        )
        return aggregated_stats_secondpass

    def _find_and_append_outliers(self, aggregated_stats):
        """
        Find outliers and append the masks in the aggregated statistics or histograms.

        This method detects outliers in the aggregated statistics or histograms using the
        outlier detectors defined in the configuration. Table containing the
        aggregated statistics or histograms will be appended with the outlier masks for each
        detector and a combined outlier mask.

        Parameters
        ----------
        aggregated_stats : astropy.table.Table
            Table containing the aggregated statistics or histograms.

        """

        outlier_mask = (
            np.zeros_like(aggregated_stats["histogram"], dtype=bool)
            if "histogram" in aggregated_stats.colnames
            else np.zeros_like(aggregated_stats["n_events"], dtype=bool)
        )
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
            match the shape of the aggregated statistics or histograms.

        Returns
        -------
        numpy.ndarray
            Boolean array where each element indicates whether the corresponding
            chunk is valid (True) or faulty (False).
        """
        # Combine gain channels, including gain-selected (single-channel) data.
        if outlier_mask.ndim >= 3:
            outlier_mask = np.logical_or.reduce(outlier_mask, axis=1)
        # Calculate the fraction of faulty pixels over the camera
        faulty_pixels = (
            np.count_nonzero(outlier_mask, axis=-1) / np.shape(outlier_mask)[-1]
        )
        # Check for valid chunks if the threshold is not exceeded
        valid_chunks = faulty_pixels < self.faulty_pixels_fraction
        return valid_chunks
