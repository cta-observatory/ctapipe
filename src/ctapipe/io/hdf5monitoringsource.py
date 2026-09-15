"""
Handles reading of monitoring files
"""

import logging
import warnings
from contextlib import ExitStack

import astropy.units as u
import numpy as np
import tables
from astropy.coordinates import AltAz, SkyCoord
from astropy.table import Row, Table
from astropy.time import Time
from astropy.utils.decorators import lazyproperty

from ..containers import (
    PEDESTAL_EVENT_TYPES,
    ArrayEventContainer,
    CameraCalibrationContainer,
    CameraMonitoringContainer,
    PixelStatisticsContainer,
    StatisticsContainer,
    TelescopePointingContainer,
)
from ..core import Provenance
from ..core.traits import AstroQuantity, List, Path
from ..exceptions import InputMissing
from ..instrument import SubarrayDescription
from .astropy_helpers import read_table
from .hdf5dataformat import (
    DL0_TEL_POINTING_GROUP,
    DL1_CAMERA_COEFFICIENTS_GROUP,
    DL1_PIXEL_STATISTICS_GROUP,
)
from .metadata import read_reference_metadata
from .monitoringsource import AvailableTypes, MonitoringSource
from .monitoringtypes import MonitoringType, TelescopeMonitoringType

__all__ = ["HDF5MonitoringSource", "get_hdf5_monitoring_types"]

logger = logging.getLogger(__name__)


def get_hdf5_monitoring_types(
    h5file: tables.File | str | Path,
) -> tuple[
    AvailableTypes, dict[int, tuple[tuple[TelescopeMonitoringType, str | None], ...]]
]:
    """Return array-wide and per-telescope (type, subtype) availability.

    Currently only telescope monitoring tables are supported, so the array-wide
    availability is empty. Pixel statistics subtypes use their HDF5 group names.
    """
    telescope_data = {}
    groups = (
        (TelescopeMonitoringType.PIXEL_STATISTICS, DL1_PIXEL_STATISTICS_GROUP),
        (TelescopeMonitoringType.CAMERA_COEFFICIENTS, DL1_CAMERA_COEFFICIENTS_GROUP),
        (TelescopeMonitoringType.TELESCOPE_POINTINGS, DL0_TEL_POINTING_GROUP),
    )
    with ExitStack() as stack:
        if not isinstance(h5file, tables.File):
            h5file = stack.enter_context(tables.open_file(h5file))

        for monitoring_type, path in groups:
            if path not in h5file:
                continue
            for table in h5file.walk_nodes(path, classname="Table"):
                if not table.name.startswith("tel_"):
                    continue
                tel_id = int(table.name.removeprefix("tel_"))
                subtype = (
                    table._v_parent._v_name
                    if monitoring_type == TelescopeMonitoringType.PIXEL_STATISTICS
                    else None
                )
                telescope_data.setdefault(tel_id, []).append((monitoring_type, subtype))

        if not telescope_data:
            warnings.warn(
                f"No monitoring types found in '{h5file.filename}'.", UserWarning
            )

    return (), {tel_id: tuple(data) for tel_id, data in telescope_data.items()}


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

    timestamp_tolerance = AstroQuantity(
        default_value=1.0 * u.s,
        help="Tolerance for timestamps outside monitoring validity ranges",
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
        self._available_telescope_data = {}
        self._pixel_stats = {}
        self._pointing_interpolator = None
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

            _, telescope_data = get_hdf5_monitoring_types(open_file)
            telescope_data = {
                tel_id: data
                for tel_id, data in telescope_data.items()
                if tel_id in self.subarray.tel
            }
            for tel_id, data in telescope_data.items():
                available = self._available_telescope_data.get(tel_id, ())
                overlapping = set(data).intersection(available)
                if overlapping:
                    msg = (
                        f"File '{file}' contains monitoring data {overlapping} for "
                        f"telescope {tel_id} that are already present in previously "
                        "processed files. This may indicate duplicate or overlapping "
                        "monitoring data."
                    )
                    self.log.warning(msg)
                    warnings.warn(msg, UserWarning)
                self._available_telescope_data[tel_id] = tuple(
                    dict.fromkeys((*available, *data))
                )

        self._process_pixel_statistics(file, telescope_data)
        self._process_camera_coefficients(file, telescope_data)
        self._process_telescope_pointings(file, telescope_data)

    def _process_pixel_statistics(self, file, telescope_data):
        """Process the pixel statistics available for each telescope in this file."""
        from ..monitoring import (
            FlatfieldImageInterpolator,
            FlatfieldPeakTimeInterpolator,
            PedestalImageInterpolator,
        )

        _interpolators = {
            "flatfield_image": FlatfieldImageInterpolator,
            "flatfield_peak_time": FlatfieldPeakTimeInterpolator,
        }
        for event_type in PEDESTAL_EVENT_TYPES:
            _interpolators[f"{event_type.name.lower()}_image"] = (
                PedestalImageInterpolator
            )

        for tel_id, data in telescope_data.items():
            for monitoring_type, name in data:
                if monitoring_type != TelescopeMonitoringType.PIXEL_STATISTICS:
                    continue

                if name not in _interpolators:
                    self.log.info(
                        f"Skipping unsupported pixel statistics subtype '{name}'"
                    )
                    continue

                if name not in self._pixel_stats:
                    interpolator = _interpolators[name](parent=self)
                    self._pixel_stats[name] = interpolator

                h5path = f"{DL1_PIXEL_STATISTICS_GROUP}/{name}/tel_{tel_id:03d}"
                table = read_table(file, h5path)
                for col in ("mean", "median", "std"):
                    table[col][table["outlier_mask"].data] = np.nan
                self._pixel_statistics.setdefault(tel_id, {})[name] = table
                self._pixel_stats[name].add_table(tel_id, table)

    def _process_camera_coefficients(self, file, telescope_data):
        """Process camera coefficients monitoring data."""
        # Read the tables from the monitoring file
        for tel_id, data in telescope_data.items():
            if (TelescopeMonitoringType.CAMERA_COEFFICIENTS, None) not in data:
                continue
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

    def _process_telescope_pointings(self, file, telescope_data):
        """Process telescope pointing monitoring data."""
        from ..monitoring import PointingInterpolator

        # Instantiate the pointing interpolator
        if self._pointing_interpolator is None:
            self._pointing_interpolator = PointingInterpolator()

        # Read the pointing data from the file
        for tel_id, data in telescope_data.items():
            if (TelescopeMonitoringType.TELESCOPE_POINTINGS, None) not in data:
                continue
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

    @property
    def available_data(self) -> AvailableTypes:
        return ()

    @property
    def available_telescope_data(
        self,
    ) -> dict[int, tuple[tuple[TelescopeMonitoringType, str | None], ...]]:
        return self._available_telescope_data.copy()

    @lazyproperty
    def has_pixel_statistics(self):
        """
        True for files that contain pixel statistics
        """
        return bool(self._pixel_statistics)

    @lazyproperty
    def has_camera_coefficients(self):
        """
        True for files that contain camera calibration coefficients
        """
        return bool(self._camera_coefficients)

    @lazyproperty
    def has_pointings(self):
        """
        True for files that contain pointing information
        """
        return bool(self._telescope_pointings)

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
        self, monitoring_type: MonitoringType, subtype: str | None = None
    ) -> Table:
        raise KeyError(
            f"Monitoring data {(monitoring_type, subtype)} not available in this source."
        )

    def get_values(
        self, time: Time, monitoring_type: MonitoringType, subtype: str | None = None
    ):
        raise KeyError(
            f"Monitoring data {(monitoring_type, subtype)} not available in this source."
        )

    def _check_telescope_data(self, tel_id, monitoring_type, subtype):
        available = self._available_telescope_data.get(tel_id, ())
        if (monitoring_type, subtype) in available:
            return
        if monitoring_type == TelescopeMonitoringType.PIXEL_STATISTICS:
            subtypes = [name for kind, name in available if kind == monitoring_type]
            if subtypes:
                message = (
                    "subtype parameter is required for PIXEL_STATISTICS."
                    if subtype is None
                    else f"Unknown subtype '{subtype}' for PIXEL_STATISTICS."
                )
                raise KeyError(f"{message} Available subtypes: {subtypes}")
        raise KeyError(
            f"Monitoring data {(monitoring_type, subtype)} not available for telescope "
            f"{tel_id}. Available data: {available}"
        )

    def get_telescope_table(
        self,
        tel_id: int,
        monitoring_type: TelescopeMonitoringType,
        subtype: str | None = None,
    ) -> Table:
        self._check_telescope_data(tel_id, monitoring_type, subtype)
        if monitoring_type == TelescopeMonitoringType.PIXEL_STATISTICS:
            return self._pixel_statistics[tel_id][subtype]
        elif monitoring_type == TelescopeMonitoringType.CAMERA_COEFFICIENTS:
            return self._camera_coefficients[tel_id]
        elif monitoring_type == TelescopeMonitoringType.TELESCOPE_POINTINGS:
            return self._telescope_pointings[tel_id]

    def _get_telescope_pointing_values(
        self,
        tel_id: int,
        time: Time,
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
        alt, az = self._pointing_interpolator(tel_id, time)
        # Get individual telescope location for proper AltAz frame
        location = self.subarray.tel_earth_locations[tel_id]
        # This is a naive SkyCoord frame that requires meteorological data
        # for proper transformations to the celestial coordinates.
        return SkyCoord(
            alt=alt,
            az=az,
            frame=AltAz(obstime=time, location=location),
        )

    def _get_camera_coefficients_values(self, tel_id: int, time: Time) -> dict:
        """
        Get camera coefficients values for a given telescope and time.

        Parameters
        ----------
        tel_id : int
            Telescope ID
        time : astropy.time.Time
            Target timestamp

        Returns
        -------
        dict[str, astropy.units.Quantity | numpy.ndarray | bool]
            Dictionary with camera coefficient data where keys are column names
            (time, factor, pedestal_offset, time_shift, outlier_mask, is_valid)
            and values are Quantity objects with appropriate units or arrays
        """
        # For simulation, use first entry if time is None
        if self.is_simulation and time is None:
            first_row = self._camera_coefficients[tel_id][0]
            return dict(zip(first_row.colnames, first_row))
        return self._get_table_rows(self._camera_coefficients[tel_id], time)

    def _get_pixel_statistics_values(
        self,
        tel_id: int,
        time: Time,
        subtype: str | None,
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

        Returns
        -------
        dict[str, astropy.units.Quantity | numpy.ndarray]
            Dictionary with pixel statistics data where keys are column names
            (mean, median, std) and values are Quantity objects or arrays.
        """
        interpolator = self._pixel_stats[subtype]
        # For simulation, use first entry if time is None
        if self.is_simulation and time is None:
            time = Time(
                self._pixel_statistics[tel_id][subtype]["time_start"][0],
                format="mjd",
            )
        return interpolator(tel_id, time, self.timestamp_tolerance)

    def get_telescope_values(
        self,
        tel_id: int,
        time: Time,
        monitoring_type: TelescopeMonitoringType,
        subtype: str | None = None,
    ):
        self._check_telescope_data(tel_id, monitoring_type, subtype)
        if monitoring_type == TelescopeMonitoringType.TELESCOPE_POINTINGS:
            return self._get_telescope_pointing_values(tel_id, time)
        elif monitoring_type == TelescopeMonitoringType.CAMERA_COEFFICIENTS:
            return self._get_camera_coefficients_values(tel_id, time)
        elif monitoring_type == TelescopeMonitoringType.PIXEL_STATISTICS:
            return self._get_pixel_statistics_values(tel_id, time, subtype)

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
            if tel_id in self._telescope_pointings and not self.is_simulation:
                event.monitoring.tel[
                    tel_id
                ].pointing = self.get_telescope_pointing_container(
                    tel_id, event.trigger.time
                )

    def get_telescope_pointing_container(
        self, tel_id: int, time: Time
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
        skycoord = self.get_telescope_values(
            tel_id, time, TelescopeMonitoringType.TELESCOPE_POINTINGS
        )
        return TelescopePointingContainer(altitude=skycoord.alt, azimuth=skycoord.az)

    def get_camera_monitoring_container(
        self,
        tel_id: int,
        time: Time = None,
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
        if tel_id in self._pixel_statistics:
            # Fill the the camera monitoring container with the pixel statistics
            pixel_stats_container = PixelStatisticsContainer()
            for name in self._pixel_statistics[tel_id]:
                stats_data = self.get_telescope_values(
                    tel_id,
                    time,
                    TelescopeMonitoringType.PIXEL_STATISTICS,
                    subtype=name,
                )
                # Map any pedestal name to the container field name (unique for pedestal)
                container_name = "pedestal_image" if "pedestal_image" in name else name
                pixel_stats_container[container_name] = StatisticsContainer(
                    mean=stats_data["mean"],
                    median=stats_data["median"],
                    std=stats_data["std"],
                )
            cam_mon_container["pixel_statistics"] = pixel_stats_container
        if tel_id in self._camera_coefficients:
            table_rows = self.get_telescope_values(
                tel_id,
                time,
                TelescopeMonitoringType.CAMERA_COEFFICIENTS,
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

    def _get_table_rows(self, table: Table, time: Time) -> dict:
        """
        Retrieve the rows of the table that corresponds to the target time.

        Parameters
        ----------
        time : astropy.time.Time
            Target timestamp(s) to find the interval.
        table : astropy.table.Table
            Table containing ordered timestamp data.

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
        tolerance_mjd = self.timestamp_tolerance.to_value("day")
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
                        f"the 'timestamp_tolerance' (currently set to '{self.timestamp_tolerance}')."
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
