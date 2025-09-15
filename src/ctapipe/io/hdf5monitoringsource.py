"""
Handles reading of monitoring files
"""

import logging
import pathlib
import warnings
from contextlib import ExitStack

import astropy
import astropy.units as u
import numpy as np
import tables
from astropy.table import Row
from astropy.utils.decorators import lazyproperty

from ..containers import (
    ArrayEventContainer,
    CameraCalibrationContainer,
    CameraMonitoringContainer,
    PixelStatisticsContainer,
    StatisticsContainer,
    TelescopePointingContainer,
)
from ..core import Provenance, ToolConfigurationError
from ..core.traits import CaselessStrEnum, List, Path
from ..instrument import SubarrayDescription
from .astropy_helpers import read_table
from .hdf5dataformat import (
    DL0_TEL_POINTING_GROUP,
    DL1_CAMERA_COEFFICIENTS_GROUP,
    DL1_PIXEL_STATISTICS_GROUP,
    DL1_TEL_CALIBRATION_GROUP,
)
from .metadata import read_reference_metadata
from .monitoringsource import MonitoringSource
from .monitoringtypes import MonitoringTypes

__all__ = ["HDF5MonitoringSource"]

logger = logging.getLogger(__name__)


def get_hdf5_monitoring_types(
    h5file: tables.File | str | Path,
) -> tuple[MonitoringTypes]:
    """Get the monitoring types present in the hdf5 file"""

    with ExitStack() as stack:
        if not isinstance(h5file, tables.File):
            h5file = stack.enter_context(tables.open_file(h5file))

        try:
            calibration_group = h5file.get_node(DL1_TEL_CALIBRATION_GROUP)
            # Iterate over enum values of MonitoringTypes
            monitoring_types = [
                monitoring_type
                for monitoring_type in [
                    MonitoringTypes.PIXEL_STATISTICS,
                    MonitoringTypes.CAMERA_COEFFICIENTS,
                ]
                if monitoring_type.value in calibration_group
            ]
            # TODO: Simplify once backwards compatibility is not needed anymore
            # Check for telescope pointing
            if DL0_TEL_POINTING_GROUP in h5file.root:
                monitoring_types.append(MonitoringTypes.TELESCOPE_POINTINGS)

        except (KeyError, tables.NoSuchNodeError):
            # TODO: Simplify once backwards compatibility is not needed anymore
            # Check for telescope pointing
            if DL0_TEL_POINTING_GROUP in h5file.root:
                monitoring_types = [MonitoringTypes.TELESCOPE_POINTINGS]
            else:
                # Return empty tuple if calibration group doesn't exist
                warnings.warn(
                    f"No monitoring types found in '{h5file.filename}'.", UserWarning
                )
                monitoring_types = []

    return tuple(monitoring_types)


class HDF5MonitoringSource(MonitoringSource):
    """
    Class for reading HDF5 monitoring data as a `~ctapipe.io.MonitoringSource`.

    This class provides a common interface for accessing HDF5 monitoring data
    from different monitoring types. An event following the ArrayEventContainer
    is passed to the `~ctapipe.io.HDF5MonitoringSource.fill_monitoring_container()`
    method and the different monitoring types are filled into a MonitoringContainer
    instance. See `~ctapipe.containers.MonitoringContainer` for details.

    A basic example on how to use the `~ctapipe.io.HDF5MonitoringSource`:

    >>> from ctapipe.io import SimTelEventSource, HDF5MonitoringSource
    >>> from ctapipe.utils import get_dataset_path
    >>> tel_id = 1
    >>> event_source = SimTelEventSource(
    ...    input_url="dataset://gamma_prod6_preliminary.simtel.zst",
    ...    allowed_tels={tel_id},
    ...    max_events=1,
    ...    skip_r1_calibration=True,
    ... )
    >>> file = get_dataset_path("calibpipe_camcalib_single_chunk_i0.1.0.dl1.h5")
    >>> monitoring_source = HDF5MonitoringSource(
    ...    subarray=event_source.subarray,
    ...    input_files=file,
    ... )
    >>> for event in event_source:
    ...     # Fill the event data with the monitoring container
    ...     monitoring_source.fill_monitoring_container(event)
    ...     # Print the monitoring information for the camera calibration
    ...     print(event.monitoring.tel[tel_id].camera.coefficients["time"])
    ...     print(event.monitoring.tel[tel_id].camera.coefficients["factor"])
    ...     print(event.monitoring.tel[tel_id].camera.coefficients["pedestal_offset"])
    ...     print(event.monitoring.tel[tel_id].camera.coefficients["time_shift"])
    ...     print(event.monitoring.tel[tel_id].camera.coefficients["outlier_mask"])
    ...     print(event.monitoring.tel[tel_id].camera.coefficients["is_valid"])
    40587.000000011576
    [[0.01539444 0.01501589 0.0158232  ... 0.01514254 0.01504862 0.01497081]
     [0.25207437 0.24654945 0.25933876 ... 0.24859268 0.24722679 0.24587582]]
    [[399.5        398.66666667 399.5        ... 399.25       398.41666667
      399.        ]
     [400.08333333 400.41666667 399.91666667 ... 400.25       399.5
      399.66666667]]
    [[ 0.01000023  0.1800003  -0.09000015 ... -0.12999916  0.1800003
       0.07999992]
     [ 0.2800007  -0.27000046  0.11000061 ...  0.04000092 -0.19000053
      -0.4699993 ]]
    [[False False False ... False False False]
     [False False False ... False False False]]
    True

    Attributes
    ----------
    input_files: list of Paths
        Paths to the input monitoring files.
    pixel_statistics: dict
        Dictionary to hold pixel statistics tables
    camera_coefficients: dict
        Dictionary to hold camera coefficients
    telescope_pointings: dict
        Dictionary to hold telescope pointing information

    """

    input_files = List(
        Path(exists=True, directory_ok=False),
        default_value=[],
        help="List of paths to the HDF5 input files containing monitoring data",
    ).tag(config=True)

    use_subarray_from = CaselessStrEnum(
        ["external", "internal", "arbitrary"],
        default_value="external",
        help=(
            "Set the origin of the subarray description. Options are: "
            "'external': use subarray from an external source (e.g. event source) "
            "'internal': use local subarray description from the monitoring file "
            "'arbitrary': use arbitrary (external has priority, internal second) or no subarray description"
        ),
    ).tag(config=True)

    def __init__(
        self, subarray=None, input_files=None, config=None, parent=None, **kwargs
    ):
        """
        MonitoringSource for monitoring files in the standard HDF5 data format

        Parameters:
        -----------
        subarray : SubarrayDescription or None
            Description of the subarray to use. If 'use_subarray_from' is set to
            'external', this must be provided. If set to 'internal', the subarray
            will be read from the monitoring files containing a subarray description.
            If set to 'arbitrary', the subarray description can be provided, read
            from the file if available or left as None.
        input_files : Path, or list of Paths, optional
            Path(s) of the files to load. Can be a single Path, or a list of Paths.
            These will be added to any files already configured via the configuration system.
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        parent:
            Parent from which the config is used. Mutually exclusive with config
        kwargs
        """

        super().__init__(
            subarray=subarray,
            config=config,
            parent=parent,
            **kwargs,
        )
        from ..monitoring import (
            FlatfieldImageInterpolator,
            FlatfieldPeakTimeInterpolator,
            PedestalImageInterpolator,
            PointingInterpolator,
        )

        # Check if the subarray is provided when required
        if self.use_subarray_from == "external" and subarray is None:
            raise ToolConfigurationError(
                "HDF5MonitoringSource: 'use_subarray_from' is set to 'external', but no "
                "subarray description is provided. Please provide a subarray description "
                "via the 'subarray' argument."
            )
        # Set the subarray description
        self.subarray = subarray

        # Handle additional input_files parameter - extend the existing list
        if input_files is not None:
            if isinstance(input_files, (list, tuple)):
                self.input_files.extend([pathlib.Path(f) for f in input_files])
            else:
                self.input_files.append(pathlib.Path(input_files))
        # Check if input_files list is empty
        if not self.input_files:
            raise IOError(
                "No input files provided. Please specify input files "
                "via configuration or input_files parameter."
            )
        # Log the input file paths
        for file in self.input_files:
            self.log.info("INPUT PATH = %s", str(file))
        # Initialize monitoring types and other useful attributes
        self._monitoring_types = set()
        self._is_simulation = None
        self._camera_coefficients, self._pixel_statistics, self._telescope_pointings = (
            {},
            {},
            {},
        )
        # Loop over the input files
        for file in self.input_files:
            if self.use_subarray_from == "internal":
                # Read the subarray description from the monitoring file
                subarray_from_file = SubarrayDescription.from_hdf(file)
                # Check that the available telescopes from the monitoring file
                # are a subset of the external subarray description if provided
                if self.subarray is not None and not set(
                    subarray_from_file.tel_ids
                ).issubset(set(self.subarray.tel_ids)):
                    raise ToolConfigurationError(
                        f"HDF5MonitoringSource: Available telescopes '{subarray_from_file.tel_ids}' from "
                        f"monitoring file '{file}' are not available in the external subarray description. "
                        f"Available telescopes in the external subarray are: '{self.subarray.tel_ids}'."
                    )
                # Overwrite the provided subarray description and log a debug message
                self.log.debug(
                    f"HDF5MonitoringSource: Using subarray description from monitoring file '{file}'. Subarray contains telescopes: "
                    f"{subarray_from_file.tel_ids}. Previously provided subarray description is overwritten."
                )
                self.subarray = subarray_from_file
            elif self.use_subarray_from == "arbitrary":
                # Read the subarray description from the file if possible
                try:
                    subarray_from_file = SubarrayDescription.from_hdf(file)
                    # Overwrite the provided subarray if it is not set
                    if self.subarray is None:
                        self.subarray = subarray_from_file
                except tables.exceptions.NoSuchNodeError as node_missing_error:
                    # Check if the reading failed with an expected error message or not
                    if (
                        str(node_missing_error)
                        == "group ``/`` does not have a child named ``/configuration/instrument/subarray/layout``"
                    ):
                        self.log.debug(
                            f"HDF5MonitoringSource: Monitoring file '{file}' does not contain a subarray description."
                        )
                    else:
                        raise IOError(
                            f"HDF5MonitoringSource: Unexpected ``tables.exceptions.NoSuchNodeError`` occurred: {node_missing_error} "
                            f"Could not read subarray description from monitoring file '{file}'."
                        )
            # Add the file to the provenance
            Provenance().add_input_file(
                str(file),
                role="Monitoring",
                reference_meta=read_reference_metadata(file),
            )
            with tables.open_file(file) as open_file:
                # Check if the file is a simulation and if it matches
                # the expected type from previous files.
                file_is_simulation = "simulation" in open_file.root
                if self._is_simulation is None:
                    self._is_simulation = file_is_simulation
                else:
                    if self._is_simulation != file_is_simulation:
                        raise IOError(
                            f"HDF5MonitoringSource: Inconsistent simulation flags found in "
                            f"file '{file}'. Previously processed files have "
                            f"simulation flag set to {self._is_simulation}, while "
                            f"current file has it set to {file_is_simulation}."
                        )
                # Get all the monitoring types from the file
                file_monitoring_types = get_hdf5_monitoring_types(open_file)
                # Check for overlapping monitoring types and warn
                overlapping_types = set(file_monitoring_types) & self._monitoring_types
                if overlapping_types:
                    overlapping_names = [mt.name for mt in overlapping_types]
                    msg = (
                        f"File '{file}' contains monitoring types {overlapping_names} "
                        f"that are already present in previously processed files. "
                        f"This may indicate duplicate or overlapping monitoring data."
                    )
                    self.log.warning(msg)
                    warnings.warn(msg, UserWarning)
                # Union of unique types
                self._monitoring_types.update(file_monitoring_types)
            # Read the different monitoring types from the file
            # Pixel statistics reading
            if MonitoringTypes.PIXEL_STATISTICS in file_monitoring_types:
                # Instantiate the chunk interpolators for each table
                self._pedestal_image_interpolator = PedestalImageInterpolator()
                self._flatfield_image_interpolator = FlatfieldImageInterpolator()
                self._flatfield_peak_time_interpolator = FlatfieldPeakTimeInterpolator()
                # Process the tables and interpolate the data
                for tel_id in self.subarray.tel_ids:
                    self._pixel_statistics[tel_id] = {}
                    for name, interpolator in (
                        ("sky_pedestal_image", self._pedestal_image_interpolator),
                        ("flatfield_image", self._flatfield_image_interpolator),
                        ("flatfield_peak_time", self._flatfield_peak_time_interpolator),
                    ):
                        # Read the tables from the monitoring file requiring all tables to be present
                        self._pixel_statistics[tel_id][name] = read_table(
                            file,
                            f"{DL1_PIXEL_STATISTICS_GROUP}/{name}/tel_{tel_id:03d}",
                        )
                        # Set outliers to NaNs
                        for col in ["mean", "median", "std"]:
                            self._pixel_statistics[tel_id][name][col][
                                self._pixel_statistics[tel_id][name][
                                    "outlier_mask"
                                ].data
                            ] = np.nan
                        # Register the table with the interpolator
                        interpolator.add_table(
                            tel_id, self._pixel_statistics[tel_id][name]
                        )
            # Camera coefficients reading
            if MonitoringTypes.CAMERA_COEFFICIENTS in file_monitoring_types:
                # Read the tables from the monitoring file requiring all tables to be present
                for tel_id in self.subarray.tel_ids:
                    self._camera_coefficients[tel_id] = read_table(
                        file,
                        f"{DL1_CAMERA_COEFFICIENTS_GROUP}/tel_{tel_id:03d}",
                    )
                # Convert time column to MJD
                self._camera_coefficients[tel_id]["time"] = self._camera_coefficients[
                    tel_id
                ]["time"].to_value("mjd")
                # Add index for the retrieval later on
                self._camera_coefficients[tel_id].add_index("time")

            # Telescope pointings reading
            if (
                MonitoringTypes.TELESCOPE_POINTINGS in file_monitoring_types
                and self.is_simulation
            ):
                msg = (
                    "HDF5MonitoringSource: Telescope pointings are available, but will be ignored. "
                    f"The monitoring file '{file}' is from simulated data."
                )
                self.log.warning(msg)
                warnings.warn(msg, UserWarning)
            if (
                MonitoringTypes.TELESCOPE_POINTINGS in file_monitoring_types
                and not self.is_simulation
            ):
                # Instantiate the pointing interpolator
                self._pointing_interpolator = PointingInterpolator()
                # Read the pointing data from the file to have the telescope pointings as a property
                for tel_id in self.subarray.tel_ids:
                    self._telescope_pointings[tel_id] = read_table(
                        file,
                        f"{DL0_TEL_POINTING_GROUP}/tel_{tel_id:03d}",
                    )
                    # Register the table with the pointing interpolator
                    self._pointing_interpolator.add_table(
                        tel_id, self._telescope_pointings[tel_id]
                    )

        # Break with an NotImplementedError if no subarray description is provided in 'arbitrary' mode.
        if self.subarray is None:
            raise NotImplementedError(
                "HDF5MonitoringSource: 'use_subarray_from' is set to 'arbitrary', but no "
                "subarray description is provided and the provided monitoring files do not "
                "contain a subarray description. Currently no functionality in the HDF5MonitoringSource "
                "is implemented to process monitoring data without a subarray description."
            )

    @property
    def is_simulation(self):
        """
        True for files with a simulation group at the root of the file.
        """
        return self._is_simulation

    @lazyproperty
    def monitoring_types(self):
        return self._monitoring_types

    @lazyproperty
    def has_pixel_statistics(self):
        """
        True for files that contain pixel statistics
        """
        return MonitoringTypes.PIXEL_STATISTICS in self.monitoring_types

    @lazyproperty
    def has_camera_coefficients(self):
        """
        True for files that contain camera calibration coefficients
        """
        return MonitoringTypes.CAMERA_COEFFICIENTS in self.monitoring_types

    @lazyproperty
    def has_pointings(self):
        """
        True for files that contain pointing information
        """
        return MonitoringTypes.TELESCOPE_POINTINGS in self.monitoring_types

    @property
    def camera_coefficients(self):
        return self._camera_coefficients

    @property
    def pixel_statistics(self):
        return self._pixel_statistics

    @property
    def telescope_pointings(self):
        return self._telescope_pointings

    def fill_monitoring_container(self, event: ArrayEventContainer):
        """
        Fill the monitoring container for a given event.

        Parameters
        ----------
        event : ArrayEventContainer
            The event to fill the monitoring container for.
        """
        # Fill the monitoring container for the event
        for tel_id in self.subarray.tel_ids:
            if self.is_simulation:
                event.monitoring.tel[
                    tel_id
                ].camera = self.get_camera_monitoring_container(tel_id)
            else:
                event.monitoring.tel[
                    tel_id
                ].camera = self.get_camera_monitoring_container(
                    tel_id, event.trigger.time
                )
            # Only overwrite the telescope pointings for observation data
            if self.has_pointings and not self.is_simulation:
                event.monitoring.tel[
                    tel_id
                ].pointing = self.get_telescope_pointing_container(
                    tel_id, event.trigger.time
                )

    def get_telescope_pointing_container(
        self, tel_id: int, time: astropy.time.Time
    ) -> TelescopePointingContainer:
        """
        Get the telescope pointing container for a given telescope ID and time.

        Parameters
        ----------
        tel_id : int
            The telescope ID to retrieve the monitoring data for.
        time : astropy.time.Time
            Target timestamp to find the telescope pointing data for.

        Returns
        -------
        TelescopePointingContainer
            The telescope pointing container.
        """
        alt, az = self._pointing_interpolator(tel_id, time)
        return TelescopePointingContainer(altitude=alt, azimuth=az)

    def get_camera_monitoring_container(
        self,
        tel_id: int,
        time: astropy.time.Time = None,
        timestamp_tolerance: u.Quantity = 0.0 * u.s,
    ) -> CameraMonitoringContainer:
        """
        Retrieve the camera monitoring container with interpolated data.

        Parameters
        ----------
        tel_id : int
            The telescope ID to retrieve the monitoring data for.
        time : astropy.time.Time or None
            Optional target timestamp(s) to find the camera monitoring data for. The target
            timestamp(s) are required to interpolate the monitoring data of observation.
            For monitoring data of simulation, the first entry of the monitoring data is typically
            used if no timestamp is provided.
        timestamp_tolerance : astropy.units.Quantity
            Time difference to consider two timestamps equal. Default is 0 seconds.

        Returns
        -------
        CameraMonitoringContainer
            The camera monitoring container.
        """
        if not self.is_simulation and time is None:
            raise ValueError(
                "Function argument 'time' must be provided for monitoring data from real observations."
            )
        if self.is_simulation and time is not None:
            msg = (
                "The function argument 'time' is provided, but the monitoring source is of simulated data. "
                "In simulations, we typically use the first entry of the monitoring data by not providing a timestamp. "
                "There is no proper time definition in simulated observing blocks. Besides, the simulation toolkit is not "
                "varying the observation conditions, e.g. raising pedestal noise level, within a given simulation run."
            )
            self.log.warning(msg)
            warnings.warn(msg, UserWarning)

        cam_mon_container = CameraMonitoringContainer()
        if self.has_pixel_statistics:
            # Fill the the camera monitoring container with the pixel statistics
            pixel_stats_container = PixelStatisticsContainer()
            for name, interpolator in (
                ("sky_pedestal_image", self._pedestal_image_interpolator),
                ("flatfield_image", self._flatfield_image_interpolator),
                ("flatfield_peak_time", self._flatfield_peak_time_interpolator),
            ):
                # Set the timestamp to the first entry if it is not provided
                # and monitoring is from simulation.
                if self.is_simulation and time is None:
                    time = self._pixel_statistics[tel_id][name]["time_start"][0]

                stats_data = interpolator(tel_id, time, timestamp_tolerance)
                pixel_stats_container[name] = StatisticsContainer(
                    mean=stats_data["mean"],
                    median=stats_data["median"],
                    std=stats_data["std"],
                )
            cam_mon_container["pixel_statistics"] = pixel_stats_container
        if self.has_camera_coefficients:
            # Set the timestamp to the first entry if it is not provided
            # and monitoring is from simulation.
            if self.is_simulation and time is None:
                time = self._camera_coefficients[tel_id]["time"][0]
            table_rows = self._get_table_rows(
                self._camera_coefficients[tel_id], time, timestamp_tolerance
            )
            cam_mon_container["coefficients"] = CameraCalibrationContainer(
                time=table_rows["time"],
                pedestal_offset=table_rows["pedestal_offset"],
                factor=table_rows["factor"],
                time_shift=table_rows["time_shift"],
                outlier_mask=table_rows["outlier_mask"],
                is_valid=table_rows["is_valid"],
            )
        return cam_mon_container

    def _get_table_rows(
        self,
        table: astropy.table.Table,
        time: astropy.time.Time,
        timestamp_tolerance: u.Quantity = 0.0 * u.s,
    ) -> dict:
        """
        Retrieve the rows of the table that corresponds to the target time.

        Parameters
        ----------
        time : astropy.time.Time
            Target timestamp(s) to find the interval.
        table : astropy.table.Table
            Table containing ordered timestamp data.
        timestamp_tolerance : astropy.units.Quantity
            Time difference in seconds to consider two timestamps equal. Default is 0s.

        Returns
        -------
        table_rows : dict
            Dictionary containing the column of the original input table as keys and
            the corresponding data for the requested time(s) as values.
        """

        mjd_times = np.atleast_1d(time.to_value("mjd"))
        table_times = table["time"]
        # Convert timestamp tolerance to MJD days
        tolerance_mjd = timestamp_tolerance.to_value("day")
        # Find the index of the closest preceding start time
        preceding_indices = np.searchsorted(table_times, mjd_times, side="right") - 1

        time_idx = []
        for mjd, preceding_index in zip(mjd_times, preceding_indices):
            # Check if the requested time is before the first chunk
            if preceding_index < 0:
                # If the time is before the first chunk and not within tolerance, break
                if (table_times[0] - tolerance_mjd) > mjd:
                    raise ValueError(
                        f"Out of bounds: Requested timestamp '{mjd} MJD' is before the "
                        f"validity start '{table['time'][0]} MJD' (first entry in the table). "
                        f"Please provide a timestamp within the validity range or increase "
                        f"the 'timestamp_tolerance' (currently set to '{timestamp_tolerance}')."
                    )
                else:
                    # Use the first chunk since it's within tolerance
                    preceding_index = 0
            # Check upper bounds when requested timestamp is after the last entry
            if preceding_index >= len(table) - 1:
                time_idx.append(table["time"][-1])
                continue
            time_idx.append(table["time"][preceding_index])
        # Get table row(s) and convert to dictionary
        table_rows = table.loc[time_idx]
        if len(time_idx) == 1:
            table_dict = (
                {col: table_rows[col] for col in table_rows.colnames}
                if isinstance(table_rows, Row)
                else {col: table_rows[col][0] for col in table_rows.colnames}
            )
        else:
            table_dict = {col: table_rows[col].data for col in table_rows.colnames}
        return table_dict
