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
    Base component for calibration calculators.

    This class provides the foundational methods and attributes for
    calculating camera-related monitoring data. It is designed
    to be extended by specific calibration calculators that implement
    the required methods for different types of calibration.

    Attributes
    ----------
    stats_aggregator_type : ctapipe.core.traits.TelescopeParameter
        The type of StatisticsAggregator to be used for aggregating statistics.
    outlier_detector_list : list of dict
        List of dictionaries containing the apply to, the name of the OutlierDetector subclass to be used, and the validity range of the detector.
    calibration_type : ctapipe.core.traits.CaselessStrEnum
        The type of calibration (e.g., pedestal, flatfield, time_calibration) which is needed to properly store the monitoring data.
    output_path : ctapipe.core.traits.Path
        The output filename where the calibration data will be stored.
    overwrite : ctapipe.core.traits.Bool
        Whether to overwrite the output file if it exists.
    """

    stats_aggregator_type = TelescopeParameter(
        trait=ComponentName(
            StatisticsAggregator, default_value="SigmaClippingAggregator"
        ),
        default_value="SigmaClippingAggregator",
        help="Name of the StatisticsAggregator subclass to be used.",
    ).tag(config=True)

    outlier_detector_list = List(
        trait=Dict,
        default_value=None,
        allow_none=True,
        help=(
            "List of dicts containing the name of the OutlierDetector subclass to be used, "
            "the aggregated value to which the detector should be applied, "
            "and the validity range of the detector."
        ),
    ).tag(config=True)

    calibration_type = CaselessStrEnum(
        ["pedestal", "flatfield", "time_calibration"],
        allow_none=False,
        help="Set type of calibration which is needed to properly store the monitoring data",
    ).tag(config=True)

    output_path = Path(
        help="Output filename", default_value=pathlib.Path("monitoring.camcalib.h5")
    ).tag(config=True)

    overwrite = Bool(help="Overwrite output file if it exists").tag(config=True)

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
            The ``StatisticsAggregator`` to use. If None, the default via the
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
        if self.outlier_detector_list is not None:
            for outlier_detector in self.outlier_detector_list:
                self.outlier_detectors[outlier_detector["apply_to"]] = (
                    OutlierDetector.from_name(
                        name=outlier_detector["name"],
                        validity_range=outlier_detector["validity_range"],
                        subarray=self.subarray,
                        parent=self,
                    )
                )

    @abstractmethod
    def __call__(self, input_url, tel_id, col_name):
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

    This class inherits from CalibrationCalculator and is responsible for
    calculating various statistics from calibration events, such as pedestal
    and flat-field data. It reads the data, aggregates statistics, detects
    outliers, handles faulty data chunks, and stores the monitoring data.
    The default option is to conduct only one pass over the data with non-overlapping
    chunks, while overlapping chunks can be set by the ``chunk_shift`` parameter.
    Two passes over the data, set via the ``second_pass``-flag, can be conducted
    with a refined shift of the chunk in regions of trouble with a high percentage
    of faulty pixels exceeding the threshold ``faulty_pixels_threshold``. 
    """

    chunk_shift = Int(
        default_value=None,
        allow_none=True,
        help=(
            "Number of samples to shift the aggregation chunk for the "
            "calculation of the statistical values. If second_pass is set, "
            "the first pass is conducted without overlapping chunks (chunk_shift=None) "
            "and the second pass with a refined shift of the chunk in regions of trouble."
        ),
    ).tag(config=True)

    second_pass = Bool(
        default_value=False,
        help=(
            "Set whether to conduct a second pass over the data "
            "with a refined shift of the chunk in regions of trouble."
        ),
    ).tag(config=True)

    faulty_pixels_threshold = Float(
        default_value=10.0,
        allow_none=True,
        help=(
            "Threshold in percentage of faulty pixels over the camera "
            "to conduct second pass with a refined shift of the chunk "
            "in regions of trouble."
        ),
    ).tag(config=True)

    def __call__(
        self,
        input_url,
        tel_id,
        col_name="image",
    ):

        # Read the whole dl1-like images of pedestal and flat-field data with the ``TableLoader``
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

        # Check if the chunk_shift is set for second pass mode
        if self.second_pass and self.chunk_shift is None:
            raise ValueError(
                "chunk_shift must be set if second pass over the data is selected"
            )

        # Get the aggregator
        aggregator = self.stats_aggregator[self.stats_aggregator_type.tel[tel_id]]
        # Pass through the whole provided dl1 data
        if self.second_pass:
            aggregated_stats = aggregator(
                table=dl1_table[tel_id], col_name=col_name, chunk_shift=None
            )
        else:
            aggregated_stats = aggregator(
                table=dl1_table[tel_id], col_name=col_name, chunk_shift=self.chunk_shift
            )
        # Detect faulty pixels with mutiple instances of ``OutlierDetector``
        outlier_mask = np.zeros_like(aggregated_stats[0]["mean"], dtype=bool)
        for aggregated_val, outlier_detector in self.outlier_detectors.items():
            outlier_mask = np.logical_or(
                outlier_mask,
                outlier_detector(aggregated_stats[aggregated_val]),
            )
        # Add the outlier mask to the aggregated statistics
        aggregated_stats["outlier_mask"] = outlier_mask

        # Conduct a second pass over the data
        if self.second_pass:
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
            ) * 100.0

            # Check for faulty chunks if the threshold is exceeded
            faulty_chunks = faulty_pixels_percentage > self.faulty_pixels_threshold
            if np.any(faulty_chunks):
                chunk_size = aggregated_stats["n_events"][0]
                faulty_chunks_indices = np.where(faulty_chunks)[0]
                for index in faulty_chunks_indices:
                    # Log information of the faulty chunks
                    self.log.warning(
                        f"Faulty chunk ({int(faulty_pixels_percentage[index]*100.0)}% of the camera unavailable) detected in the first pass: time_start={aggregated_stats['time_start'][index]}; time_end={aggregated_stats['time_end'][index]}"
                    )
                    # Calculate the start of the slice based
                    slice_start = (
                        chunk_size * index
                        if index - 1 in faulty_chunks_indices
                        else chunk_size * (index - 1)
                    )
                    # Set the start of the slice to the first element of the dl1 table if out of bound
                    # and add one ``chunk_shift``.
                    slice_start = max(0, slice_start) + self.chunk_shift
                    # Set the end of the slice to the last element of the dl1 table if out of bound
                    # and subtract one ``chunk_shift``.
                    slice_end = min(
                        len(dl1_table[tel_id]) - 1, chunk_size * (index + 2)
                    ) - (self.chunk_shift - 1)
                    # Slice the dl1 table according to the previously caluclated start and end.
                    dl1_table_sliced = dl1_table[tel_id][slice_start:slice_end]

                    # Run the stats aggregator on the sliced dl1 table with a chunk_shift
                    # to sample the period of trouble (carflashes etc.) as effectively as possible.
                    aggregated_stats_secondpass = aggregator(
                        table=dl1_table_sliced,
                        col_name=col_name,
                        chunk_shift=self.chunk_shift,
                    )

                    # Detect faulty pixels with mutiple instances of OutlierDetector of the second pass
                    outlier_mask_secondpass = np.zeros_like(
                        aggregated_stats_secondpass[0]["mean"], dtype=bool
                    )
                    for (
                        aggregated_val,
                        outlier_detector,
                    ) in self.outlier_detectors.items():
                        outlier_mask_secondpass = np.logical_or(
                            outlier_mask_secondpass,
                            outlier_detector(
                                aggregated_stats_secondpass[aggregated_val]
                            ),
                        )
                    # Add the outlier mask to the aggregated statistics
                    aggregated_stats_secondpass["outlier_mask"] = outlier_mask_secondpass

                    # Stack the aggregated statistics of the second pass to the first pass
                    aggregated_stats = vstack(
                        [aggregated_stats, aggregated_stats_secondpass]
                    )
                    # Sort the aggregated statistics based on the starting time
                    aggregated_stats.sort(["time_start"])
            else:
                self.log.info(
                    "No faulty chunks detected in the first pass. The second pass with a finer chunk shift is not executed."
                )

        # Write the aggregated statistics and their outlier mask to the output file
        write_table(
            aggregated_stats,
            self.output_path,
            f"{self.group[self.calibration_type]}/tel_{tel_id:03d}",
            overwrite=self.overwrite,
        )
