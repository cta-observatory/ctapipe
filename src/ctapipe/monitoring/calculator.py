"""
Definition of the ``CalibrationCalculator`` classes, providing all steps needed to
calculate the montoring data for the camera calibration.
"""

import pathlib
from abc import abstractmethod

import numpy as np
from astropy.table import vstack

from ctapipe.core import TelescopeComponent
from ctapipe.core.traits import (
    Bool,
    CaselessStrEnum,
    ComponentName,
    Dict,
    Float,
    Int,
    List,
    Path,
    TelescopeParameter,
)
from ctapipe.io import write_table
from ctapipe.io.tableloader import TableLoader
from ctapipe.monitoring.aggregator import StatisticsAggregator
from ctapipe.monitoring.outlier import OutlierDetector

__all__ = [
    "CalibrationCalculator",
    "StatisticsCalculator",
]

PEDESTAL_GROUP = "/dl0/monitoring/telescope/pedestal"
FLATFIELD_GROUP = "/dl0/monitoring/telescope/flatfield"
TIMECALIB_GROUP = "/dl0/monitoring/telescope/time_calibration"


class CalibrationCalculator(TelescopeComponent):
    """
    Base component for various calibration calculators

    Attributes
    ----------
    stats_aggregator: str
        The name of the StatisticsAggregator subclass to be used to aggregate the statistics
    """

    stats_aggregator_type = TelescopeParameter(
        trait=ComponentName(
            StatisticsAggregator, default_value="SigmaClippingAggregator"
        ),
        default_value="SigmaClippingAggregator",
        help="Name of the StatisticsAggregator subclass to be used.",
    ).tag(config=True)

    outlier_detector_type = List(
        trait=Dict,
        default_value=None,
        allow_none=True,
        help=(
            "List of dictionaries containing the apply to, the name of the OutlierDetector subclass to be used, and the validity range of the detector."
        ),
    ).tag(config=True)

    calibration_type = CaselessStrEnum(
        ["pedestal", "flatfield", "time_calibration"],
        allow_none=False,
        help="Set type of calibration which is needed to properly store the monitoring data",
    ).tag(config=True)

    output_path = Path(
        help="output filename", default_value=pathlib.Path("monitoring.camcalib.h5")
    ).tag(config=True)

    overwrite = Bool(help="overwrite output file if it exists").tag(config=True)

    def __init__(
        self,
        subarray,
        config=None,
        parent=None,
        stats_aggregator=None,
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
        stats_aggregator: ctapipe.monitoring.aggregator.StatisticsAggregator
            The StatisticsAggregator to use. If None, the default via the
            configuration system will be constructed.
        """
        super().__init__(subarray=subarray, config=config, parent=parent, **kwargs)
        self.subarray = subarray

        self.group = {
            "pedestal": PEDESTAL_GROUP,
            "flatfield": FLATFIELD_GROUP,
            "time_calibration": TIMECALIB_GROUP,
        }

        # Initialize the instances of StatisticsAggregator
        self.stats_aggregator = {}
        if stats_aggregator is None:
            for _, _, name in self.stats_aggregator_type:
                self.stats_aggregator[name] = StatisticsAggregator.from_name(
                    name, subarray=self.subarray, parent=self
                )
        else:
            name = stats_aggregator.__class__.__name__
            self.stats_aggregator_type = [("type", "*", name)]
            self.stats_aggregator[name] = stats_aggregator

        # Initialize the instances of OutlierDetector
        self.outlier_detectors = {}
        if self.outlier_detector_type is not None:
            for outlier_detector in self.outlier_detector_type:
                self.outlier_detectors[outlier_detector["apply_to"]] = (
                    OutlierDetector.from_name(
                        name=outlier_detector["name"],
                        validity_range=outlier_detector["validity_range"],
                        subarray=self.subarray,
                        parent=self,
                    )
                )

    @abstractmethod
    def __call__(self, input_url, tel_id):
        """
        Call the relevant functions to calculate the calibration coefficients
        for a given set of events

        Parameters
        ----------
        input_url : str
            URL where the events are stored from which the calibration coefficients
            are to be calculated
        tel_id : int
            The telescope id
        """


class StatisticsCalculator(CalibrationCalculator):
    """
    Component to calculate statistics from calibration events.
    """

    chunk_shift = Int(
        default_value=None,
        allow_none=True,
        help="Number of samples to shift the extraction chunk for the calculation of the statistical values",
    ).tag(config=True)

    two_pass = Bool(default_value=False, help="overwrite output file if it exists").tag(
        config=True
    )

    faulty_pixels_threshold = Float(
        default_value=0.1,
        allow_none=True,
        help=(
            "Percentage of faulty pixels over the camera to conduct second pass with refined shift of the chunk"
        ),
    ).tag(config=True)

    def __call__(
        self,
        input_url,
        tel_id,
        col_name="image",
    ):

        # Read the whole dl1-like images of pedestal and flat-field data with the TableLoader
        input_data = TableLoader(input_url=input_url)
        dl1_table = input_data.read_telescope_events_by_id(
            telescopes=tel_id,
            dl1_images=True,
            dl1_parameters=False,
            dl1_muons=False,
            dl2=False,
            simulated=False,
            true_images=False,
            true_parameters=False,
            instrument=False,
            pointing=False,
        )

        # Check if the chunk_shift is set for two pass mode
        if self.two_pass and self.chunk_shift is None:
            raise ValueError("chunk_shift must be set for two pass mode")

        # Get the aggregator
        aggregator = self.stats_aggregator[self.stats_aggregator_type.tel[tel_id]]
        # Pass through the whole provided dl1 data
        if self.two_pass:
            self.aggregated_stats = aggregator(
                table=dl1_table[tel_id], col_name=col_name, chunk_shift=None
            )
        else:
            self.aggregated_stats = aggregator(
                table=dl1_table[tel_id], col_name=col_name, chunk_shift=self.chunk_shift
            )
        # Detect faulty pixels with mutiple instances of OutlierDetector
        outlier_mask = np.zeros_like(self.aggregated_stats[0]["mean"], dtype=bool)
        for aggregated_val, outlier_detector in self.outlier_detectors.items():
            outlier_mask = np.logical_or(
                outlier_mask,
                outlier_detector(self.aggregated_stats[aggregated_val]),
            )
        # Add the outlier mask to the aggregated statistics
        self.aggregated_stats["outlier_mask"] = outlier_mask

        if self.two_pass:
            # Check if the camera has two gain channels
            if outlier_mask.shape[1] == 2:
                # Combine the outlier mask of both gain channels
                outlier_mask = np.logical_or(
                    outlier_mask[:, 0, :],
                    outlier_mask[:, 1, :],
                )
            # Calculate the fraction of faulty pixels over the camera
            faulty_pixels_percentage = (
                np.count_nonzero(outlier_mask, axis=-1) / np.shape(outlier_mask)[-1]
            )

            # Check for faulty chunks if the threshold is exceeded
            faulty_chunks = faulty_pixels_percentage > self.faulty_pixels_threshold
            if np.any(faulty_chunks):
                faulty_chunks_indices = np.where(faulty_chunks)[0]
                for index in faulty_chunks_indices:
                    # Log information of the faulty chunks
                    self.log.warning(
                        f"Faulty chunks ({int(faulty_pixels_percentage[index]*100.0)}% of the camera unavailable) detected in the first pass: time_start={self.aggregated_stats['time_start'][index]}; time_end={self.aggregated_stats['time_end'][index]}"
                    )

                    # Slice the dl1 table according to the previously caluclated start and end.
                    slice_start, slice_end = self._get_slice_range(
                        chunk_index=index,
                        faulty_previous_chunk=(index-1 in faulty_chunks_indices),
                        dl1_table_length=len(dl1_table[tel_id]) - 1,
                    )
                    dl1_table_sliced = dl1_table[tel_id][slice_start:slice_end]

                    # Run the stats aggregator on the sliced dl1 table with a chunk_shift
                    # to sample the period of trouble (carflashes etc.) as effectively as possible.
                    aggregated_stats_secondpass = aggregator(
                        table=dl1_table_sliced,
                        col_name=col_name,
                        chunk_shift=self.chunk_shift,
                    )

                    # Detect faulty pixels with mutiple instances of OutlierDetector of the second pass
                    outlier_mask = np.zeros_like(aggregated_stats_secondpass[0]["mean"], dtype=bool)
                    for aggregated_val, outlier_detector in self.outlier_detectors.items():
                        outlier_mask = np.logical_or(
                            outlier_mask,
                            outlier_detector(aggregated_stats_secondpass[aggregated_val]),
                        )
                    # Add the outlier mask to the aggregated statistics
                    aggregated_stats_secondpass["outlier_mask"] = outlier_mask

                    # Stack the aggregated statistics of the second pass to the first pass
                    self.aggregated_stats = vstack([self.aggregated_stats, aggregated_stats_secondpass])
                    # Sort the aggregated statistics based on the starting time
                    self.aggregated_stats.sort(["time_start"])
            else:
                self.log.info(
                    "No faulty chunks detected in the first pass. The second pass with a finer chunk shift is not executed."
                )

        # Write the aggregated statistics and their outlier mask to the output file
        write_table(
            self.aggregated_stats,
            self.output_path,
            f"{self.group[self.calibration_type]}/tel_{tel_id:03d}",
            overwrite=self.overwrite,
        )

    def _get_slice_range(
        self,
        chunk_index,
        faulty_previous_chunk,
        dl1_table_length,
    ) -> (int, int):
        """
        Calculate the start and end indices for slicing the DL1 table to be used for the second pass.

        Parameters
        ----------
        chunk_index : int
            The index of the current faulty chunk being processed.
        faulty_previous_chunk : bool
            A flag indicating if the previous chunk was faulty.
        dl1_table_length : int
            The total length of the DL1 table.

        Returns
        -------
        tuple
            A tuple containing the start and end indices for slicing the DL1 table.
        """

        # Set the start of the slice to the first element of the dl1 table
        slice_start = 0
        if chunk_index > 0:
            # Get the start of the previous chunk
            if faulty_previous_chunk:
                slice_start = np.sum(self.aggregated_stats["n_events"][:chunk_index])
            else:
                slice_start = np.sum(
                    self.aggregated_stats["n_events"][: chunk_index - 1]
                )

        # Set the end of the slice to the last element of the dl1 table
        slice_end = dl1_table_length
        if chunk_index < len(self.aggregated_stats) - 1:
            # Get the stop of the next chunk
            slice_end = np.sum(self.aggregated_stats["n_events"][: chunk_index + 2])

        # Shift the start and end of the slice by the chunk_shift
        slice_start += self.chunk_shift
        slice_end -= self.chunk_shift - 1

        return int(slice_start), int(slice_end)
