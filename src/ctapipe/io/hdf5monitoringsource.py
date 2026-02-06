"""
Handles reading of monitoring files
"""

import logging
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
from ..core import Provenance
from ..core.traits import List, Path
from ..exceptions import InputMissing
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
from .monitoringtypes import TELESCOPE_SPECIFIC_MONITORING, MonitoringType

__all__ = ["HDF5MonitoringSource", "get_hdf5_monitoring_types"]

logger = logging.getLogger(__name__)


def get_hdf5_monitoring_types(
    h5file: tables.File | str | Path,
) -> tuple[MonitoringType]:
    """Get the monitoring types present in the hdf5 file"""

    with ExitStack() as stack:
        if not isinstance(h5file, tables.File):
            h5file = stack.enter_context(tables.open_file(h5file))

        try:
            calibration_group = h5file.get_node(DL1_TEL_CALIBRATION_GROUP)
            # Iterate over enum values of MonitoringType
            monitoring_types = [
                monitoring_type
                for monitoring_type in [
                    MonitoringType.PIXEL_STATISTICS,
                    MonitoringType.CAMERA_COEFFICIENTS,
                ]
                if monitoring_type.value in calibration_group
            ]
            # TODO: Simplify once backwards compatibility is not needed anymore
            # Check for telescope pointing
            if DL0_TEL_POINTING_GROUP in h5file.root:
                monitoring_types.append(MonitoringType.TELESCOPE_POINTINGS)

        except (KeyError, tables.NoSuchNodeError):
            # TODO: Simplify once backwards compatibility is not needed anymore
            # Check for telescope pointing
            if DL0_TEL_POINTING_GROUP in h5file.root:
                monitoring_types = [MonitoringType.TELESCOPE_POINTINGS]
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
    ...    input_files=[file],
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

    def __init__(self, subarray=None, config=None, parent=None, **kwargs):
        """
        MonitoringSource for monitoring files in the standard HDF5 data format

        Parameters:
        -----------
        subarray : SubarrayDescription or None
            Optional description of the subarray. If provided, the subarray
            description should match the one from the monitoring file(s).
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

        # Check if input_files list is empty
        if not self.input_files:
            raise InputMissing(
                "No input files provided. Please specify a list of input file(s) "
                "via configuration by `--HDF5MonitoringSource.input_files` "
                "or using as an argument <input_files> in the constructor."
            )

        # Initialize attributes
        self._monitoring_types = set()
        self._is_simulation = None
        self._camera_coefficients = {}
        self._pixel_statistics = {}
        self._telescope_pointings = {}

        # Read and validate subarray descriptions
        self._read_and_validate_subarrays()

        # Process all monitoring files
        for file in self.input_files:
            self._process_single_file(file)

    def _read_and_validate_subarrays(self):
        """Read subarray descriptions from files and validate compatibility."""
        # Loop over the input files to read the subarray description and check for compatibility
        # if a subarray is already provided either externally or via a previous monitoring file.
        subarrays = ([self.subarray] if self.subarray is not None else []) + [
            SubarrayDescription.from_hdf(f) for f in self.input_files
        ]
        # Check if all subarray descriptions are compatible
        if not SubarrayDescription.check_matching_subarrays(subarrays):
            raise IOError("Incompatible subarray descriptions found in input files.")
        # Set the subarray description
        self.subarray = subarrays[0]

    def _process_single_file(self, file):
        """Process a single monitoring file."""
        # Add the file to the provenance
        Provenance().add_input_file(
            str(file),
            role="Monitoring",
            reference_meta=read_reference_metadata(file),
        )

        with tables.open_file(file) as open_file:
            # Validate simulation consistency
            # Determine if the file is from simulation.
            # First check for the presence of the simulation group.
            file_is_simulation = False
            if "simulation" in open_file.root:
                file_is_simulation = True
            else:
                # Check for metadata attribute if simulation group is not present
                if (
                    "CTA PRODUCT DATA CATEGORY" in open_file.root._v_attrs
                    and open_file.root._v_attrs["CTA PRODUCT DATA CATEGORY"] == "Sim"
                ):
                    file_is_simulation = True

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

            # Get monitoring types from the file
            file_monitoring_types = get_hdf5_monitoring_types(open_file)
            # Check for overlapping monitoring types
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
            # Update monitoring types
            self._monitoring_types.update(file_monitoring_types)

        # Process each monitoring type
        if MonitoringType.PIXEL_STATISTICS in file_monitoring_types:
            self._process_pixel_statistics(file)

        if MonitoringType.CAMERA_COEFFICIENTS in file_monitoring_types:
            self._process_camera_coefficients(file)

        if MonitoringType.TELESCOPE_POINTINGS in file_monitoring_types:
            self._process_telescope_pointings(file)

    def _process_pixel_statistics(self, file):
        """Process pixel statistics monitoring data."""
        from ..monitoring import (
            FlatfieldImageInterpolator,
            FlatfieldPeakTimeInterpolator,
            PedestalImageInterpolator,
        )

        # Open the file to check for the existence of pixel statistics tables
        with tables.open_file(file, mode="r") as h5file:
            # Iterate over pixel statistics tables to check for their existence
            self.pixel_stats_dict = {}
            for group in h5file.walk_groups(DL1_PIXEL_STATISTICS_GROUP):
                # Skip the parent group itself
                if group._v_pathname == DL1_PIXEL_STATISTICS_GROUP:
                    continue
                # Instantiate the appropriate interpolator based on the table name
                name = group._v_name
                if "pedestal_image" in name:
                    self.pixel_stats_dict[name] = PedestalImageInterpolator()
                elif "flatfield_image" in name:
                    self.pixel_stats_dict[name] = FlatfieldImageInterpolator()
                elif "flatfield_peak_time" in name:
                    self.pixel_stats_dict[name] = FlatfieldPeakTimeInterpolator()

        # Process the tables and interpolate the data
        for tel_id in self.subarray.tel_ids:
            self._pixel_statistics[tel_id] = {}

            for name, interpolator in self.pixel_stats_dict.items():
                # Read the tables from the monitoring file
                self._pixel_statistics[tel_id][name] = read_table(
                    file,
                    f"{DL1_PIXEL_STATISTICS_GROUP}/{name}/tel_{tel_id:03d}",
                )

                # Set outliers to NaNs
                for col in ["mean", "median", "std"]:
                    self._pixel_statistics[tel_id][name][col][
                        self._pixel_statistics[tel_id][name]["outlier_mask"].data
                    ] = np.nan

                # Register the table with the interpolator
                interpolator.add_table(tel_id, self._pixel_statistics[tel_id][name])

    def _process_camera_coefficients(self, file):
        """Process camera coefficients monitoring data."""
        # Read the tables from the monitoring file
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

    def _process_telescope_pointings(self, file):
        """Process telescope pointing monitoring data."""
        from ..monitoring import PointingInterpolator

        # Instantiate the pointing interpolator
        self._pointing_interpolator = PointingInterpolator()

        # Read the pointing data from the file
        for tel_id in self.subarray.tel_ids:
            self._telescope_pointings[tel_id] = read_table(
                file,
                f"{DL0_TEL_POINTING_GROUP}/tel_{tel_id:03d}",
            )

            # Register the table with the pointing interpolator
            self._pointing_interpolator.add_table(
                tel_id, self._telescope_pointings[tel_id]
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
        return MonitoringType.PIXEL_STATISTICS in self.monitoring_types

    @lazyproperty
    def has_camera_coefficients(self):
        """
        True for files that contain camera calibration coefficients
        """
        return MonitoringType.CAMERA_COEFFICIENTS in self.monitoring_types

    @lazyproperty
    def has_pointings(self):
        """
        True for files that contain pointing information
        """
        return MonitoringType.TELESCOPE_POINTINGS in self.monitoring_types

    @property
    def camera_coefficients(self):
        return self._camera_coefficients

    @property
    def pixel_statistics(self):
        return self._pixel_statistics

    @property
    def telescope_pointings(self):
        return self._telescope_pointings

    def get_table(
        self,
        monitoring_type: MonitoringType,
        tel_id: int = None,
        **kwargs,
    ):
        if monitoring_type not in self.monitoring_types:
            raise KeyError(
                f"Monitoring type {monitoring_type} not available in this source. "
                f"Available types: {self.monitoring_types}"
            )

        if monitoring_type in TELESCOPE_SPECIFIC_MONITORING and tel_id is None:
            raise TypeError(
                f"tel_id is required for {monitoring_type.name} monitoring type"
            )

        if monitoring_type == MonitoringType.PIXEL_STATISTICS:
            subtype = kwargs.get("subtype")
            if subtype is None:
                raise KeyError(
                    "subtype parameter is required for PIXEL_STATISTICS. "
                    f"Available subtypes: {list(self.pixel_stats_dict.keys())}"
                )
            if subtype not in self.pixel_stats_dict:
                raise KeyError(
                    f"Unknown subtype '{subtype}' for PIXEL_STATISTICS. "
                    f"Available subtypes: {list(self.pixel_stats_dict.keys())}"
                )
            return self._pixel_statistics[tel_id][subtype]
        elif monitoring_type == MonitoringType.CAMERA_COEFFICIENTS:
            return self._camera_coefficients[tel_id]
        elif monitoring_type == MonitoringType.TELESCOPE_POINTINGS:
            return self._telescope_pointings[tel_id]

    def _get_telescope_pointing_values(
        self,
        tel_id: int,
        time: astropy.time.Time,
    ):
        """
        Get telescope pointing values for a given telescope and time.

        Parameters
        ----------
        tel_id : int
            Telescope ID
        time : astropy.time.Time
            Target timestamp

        Returns
        -------
        astropy.coordinates.SkyCoord
            Sky coordinate with altitude and azimuth in AltAz frame
        """
        from astropy.coordinates import AltAz, SkyCoord

        alt, az = self._pointing_interpolator(tel_id, time)
        # Get individual telescope location for proper AltAz frame
        tel_index = self.subarray.tel_index_array[tel_id]
        location = self.subarray.tel_coords[tel_index].to_earth_location()
        # TODO: Get pressure from weather station data
        # Hardcoded for nominal pressure at 2200m a.s.l. (CTA North, La Palma)
        pressure = 780.0 * u.hPa
        return SkyCoord(
            alt=alt,
            az=az,
            frame=AltAz(obstime=time, location=location, pressure=pressure),
        )

    def _get_camera_coefficients_values(
        self,
        tel_id: int,
        time: astropy.time.Time,
        timestamp_tolerance: u.Quantity,
    ) -> dict:
        """
        Get camera coefficients values for a given telescope and time.

        Parameters
        ----------
        tel_id : int
            Telescope ID
        time : astropy.time.Time
            Target timestamp
        timestamp_tolerance : astropy.units.Quantity
            Time difference to consider two timestamps equal

        Returns
        -------
        dict[str, astropy.units.Quantity | numpy.ndarray | bool]
            Dictionary with camera coefficient data where keys are column names
            (time, factor, pedestal_offset, time_shift, outlier_mask, is_valid)
            and values are Quantity objects with appropriate units or arrays
        """
        # For simulation, use first entry if time is None
        if self.is_simulation and time is None:
            from astropy.time import Time

            time = Time(self._camera_coefficients[tel_id]["time"][0], format="mjd")
        return self._get_table_rows(
            self._camera_coefficients[tel_id], time, timestamp_tolerance
        )

    def _get_pixel_statistics_values(
        self,
        tel_id: int,
        time: astropy.time.Time,
        subtype: str,
        timestamp_tolerance: u.Quantity,
    ) -> dict:
        """
        Get pixel statistics values for a given telescope and time.

        Parameters
        ----------
        tel_id : int
            Telescope ID
        time : astropy.time.Time
            Target timestamp
        subtype : str
            Subtype of pixel statistics (e.g., 'pedestal_image', 'flatfield_image')
        timestamp_tolerance : astropy.units.Quantity
            Time difference to consider two timestamps equal

        Returns
        -------
        dict[str, astropy.units.Quantity | numpy.ndarray]
            Dictionary with pixel statistics data where keys are column names
            (mean, median, std) and values are Quantity objects or arrays.
        """
        if subtype is None:
            raise KeyError(
                "subtype parameter is required for PIXEL_STATISTICS. "
                f"Available subtypes: {list(self.pixel_stats_dict.keys())}"
            )
        if subtype not in self.pixel_stats_dict:
            raise KeyError(
                f"Unknown subtype '{subtype}' for PIXEL_STATISTICS. "
                f"Available subtypes: {list(self.pixel_stats_dict.keys())}"
            )
        interpolator = self.pixel_stats_dict[subtype]
        # For simulation, use first entry if time is None
        if self.is_simulation and time is None:
            from astropy.time import Time

            time = Time(
                self._pixel_statistics[tel_id][subtype]["time_start"][0],
                format="mjd",
            )
        return interpolator(tel_id, time, timestamp_tolerance)

    def get_values(
        self,
        monitoring_type: MonitoringType,
        time: astropy.time.Time,
        tel_id: int = None,
        **kwargs,
    ):
        import astropy.units as u

        if monitoring_type not in self.monitoring_types:
            raise KeyError(
                f"Monitoring type {monitoring_type} not available in this source. "
                f"Available types: {self.monitoring_types}"
            )

        if monitoring_type in TELESCOPE_SPECIFIC_MONITORING and tel_id is None:
            raise TypeError(
                f"tel_id is required for {monitoring_type.name} monitoring type"
            )

        timestamp_tolerance = kwargs.get("timestamp_tolerance", 0.0 * u.s)

        if monitoring_type == MonitoringType.TELESCOPE_POINTINGS:
            return self._get_telescope_pointing_values(tel_id, time)
        elif monitoring_type == MonitoringType.CAMERA_COEFFICIENTS:
            return self._get_camera_coefficients_values(
                tel_id, time, timestamp_tolerance
            )
        elif monitoring_type == MonitoringType.PIXEL_STATISTICS:
            subtype = kwargs.get("subtype")
            return self._get_pixel_statistics_values(
                tel_id, time, subtype, timestamp_tolerance
            )

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
            time = None if self.is_simulation else event.trigger.time
            event.monitoring.tel[tel_id].camera = self.get_camera_monitoring_container(
                tel_id, time
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
        skycoord = self.get_values(
            MonitoringType.TELESCOPE_POINTINGS, time=time, tel_id=tel_id
        )
        return TelescopePointingContainer(altitude=skycoord.alt, azimuth=skycoord.az)

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
            for name in self.pixel_stats_dict.keys():
                stats_data = self.get_values(
                    MonitoringType.PIXEL_STATISTICS,
                    time=time,
                    tel_id=tel_id,
                    subtype=name,
                    timestamp_tolerance=timestamp_tolerance,
                )
                # Map any pedestal name to the container field name (unique for pedestal)
                container_name = "pedestal_image" if "pedestal_image" in name else name
                pixel_stats_container[container_name] = StatisticsContainer(
                    mean=stats_data["mean"],
                    median=stats_data["median"],
                    std=stats_data["std"],
                )
            cam_mon_container["pixel_statistics"] = pixel_stats_container
        if self.has_camera_coefficients:
            table_rows = self.get_values(
                MonitoringType.CAMERA_COEFFICIENTS,
                time=time,
                tel_id=tel_id,
                timestamp_tolerance=timestamp_tolerance,
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
        table_rows : dict[str, astropy.units.Quantity | numpy.ndarray | Any]
            Dictionary containing the column names of the original input table as keys and
            the corresponding data (with units preserved as Quantity objects) for the
            requested time(s) as values.
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
