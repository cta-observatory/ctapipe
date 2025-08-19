"""
Handles reading of monitoring files
"""
import logging
from collections.abc import Generator
from contextlib import ExitStack

import astropy
import numpy as np
import tables
from astropy.utils.decorators import lazyproperty

from ..containers import (
    ArrayEventContainer,
    CameraCalibrationContainer,
    MonitoringCameraContainer,
    PixelStatisticsContainer,
    StatisticsContainer,
    TelescopePointingContainer,
)
from ..core import Provenance, ToolConfigurationError
from ..core.traits import Bool, Path
from ..instrument import SubarrayDescription
from ..monitoring import (
    FlatfieldImageInterpolator,
    FlatfieldPeakTimeInterpolator,
    PedestalImageInterpolator,
    PointingInterpolator,
)
from .astropy_helpers import read_table
from .metadata import read_reference_metadata
from .monitoringsource import MonitoringSource
from .monitoringtypes import MonitoringTypes

__all__ = ["HDF5MonitoringSource"]

logger = logging.getLogger(__name__)

TELESCOPE_CALIBRATION_GROUP = "/dl1/monitoring/telescope/calibration"


def get_hdf5_monitoring_types(
    h5file: tables.File | str | Path,
) -> tuple[MonitoringTypes]:
    """Get the monitoring types present in the hdf5 file"""

    with ExitStack() as stack:
        if not isinstance(h5file, tables.File):
            h5file = stack.enter_context(tables.open_file(h5file))

        try:
            calibration_group = h5file.get_node(TELESCOPE_CALIBRATION_GROUP)
            # Iterate over enum values of MonitoringTypes
            monitoring_types = [
                monitoring_type
                for monitoring_type in MonitoringTypes
                if monitoring_type.value in calibration_group
            ]
        except (KeyError, tables.NoSuchNodeError):
            # Return empty tuple if calibration group doesn't exist
            monitoring_types = []

    return tuple(monitoring_types)


class HDF5MonitoringSource(MonitoringSource):
    """
    Class for reading HDF5 monitoring data.

    This class provides a common interface for accessing HDF5 monitoring data from different monitoring types.
    TODO: Fill proper docstring.

    Attributes
    ----------
    input_url: str
        Path to the input monitoring file.
    file: tables.File
        File object
    pixel_statistics: dict
        Dictionary to hold pixel statistics tables
    camera_coefficients: astropy.table.Table
        Table to hold camera coefficients

    """

    input_url = Path(
        help="Path to the HDF5 input file containing monitoring data."
    ).tag(config=True)

    overwrite_telescope_pointings = Bool(
        False,
        help=(
            "Flag to overwrite the telescope pointing information from the monitoring data. "
            "Different EventSource implementations may already provide this information."
        ),
    ).tag(config=True)

    enforce_subarray_description = Bool(
        True,
        help=(
            "Force the reading of the subarray description "
            "from the input file if the subarray is not provided."
        ),
    ).tag(config=True)

    def __init__(
        self, subarray=None, input_url=None, config=None, parent=None, **kwargs
    ):
        """
        MonitoringSource for monitoring files in the standard HDF5 data format

        Parameters:
        -----------
        subarray : SubarrayDescription or None
            Description of the subarray to use. If not provided, the subarray
            will be read from the input file. The 'enforce_subarray_description' flag
            determines whether a subarray description is required.
        input_url : str
            Path of the file to load
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
            input_url=input_url,
            config=config,
            parent=parent,
            **kwargs,
        )

        # Check if the reading from the input file is enforced
        if self.enforce_subarray_description:
            # Read the subarray description from the file if required
            self._subarray_from_file = SubarrayDescription.read(self.input_url)
            # Overwrite the provided subarray if it is not set
            if self.subarray is None:
                self.subarray = self._subarray_from_file
            # Check if the requested telescopes are available in the file
            if not set(self.subarray.tel_ids).issubset(
                set(self._subarray_from_file.tel_ids)
            ):
                raise ToolConfigurationError(
                    f"HDF5MonitoringSource: Requested telescopes '{self.subarray.tel_ids}' are not "
                    f"available in the monitoring file '{self.input_url}'. Available telescopes "
                    f"are: '{self._subarray_from_file.tel_ids}'."
                )

        # Raise an error that the all the current monitoring types need a subarray to be defined
        if self.subarray is None and not self.enforce_subarray_description:
            raise NotImplementedError(
                "Subarray is not defined, but all implemented monitoring types "
                "of the HDF5MonitoringSource requires a defined subarray."
            )

        # Open the file and read the metadata
        self.file_ = tables.open_file(self.input_url)
        Provenance().add_input_file(
            str(self.input_url),
            role="Monitoring",
            reference_meta=read_reference_metadata(self.input_url),
        )
        # Read the different monitoring types from the file
        # Pixel statistics reading
        self._pixel_statistics = {}
        if self.has_pixel_statistics:
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
                        self.input_url,
                        f"{TELESCOPE_CALIBRATION_GROUP}/camera/pixel_statistics/{name}/tel_{tel_id:03d}",
                    )
                    if not self.is_simulation:
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
        self._camera_coefficients = {}
        if self.has_camera_coefficients:
            # Read the tables from the monitoring file requiring all tables to be present
            for tel_id in self.subarray.tel_ids:
                self._camera_coefficients[tel_id] = read_table(
                    self.input_url,
                    f"{TELESCOPE_CALIBRATION_GROUP}/camera/coefficients/tel_{tel_id:03d}",
                )
            # Convert time column to MJD
            self._camera_coefficients[tel_id]["time"] = self._camera_coefficients[
                tel_id
            ]["time"].to_value("mjd")

        # Telescope pointings reading
        self._telescope_pointings = {}
        if not self.has_pointings and self.overwrite_telescope_pointings:
            self.close()
            raise ToolConfigurationError(
                "HDF5MonitoringSource: Telescope pointings are not available in the file, "
                "but overwriting of the telescope pointings is enforced."
            )
        if self.has_pointings and self.overwrite_telescope_pointings:
            # Instantiate the pointing interpolator
            self._pointing_interpolator = PointingInterpolator()
            # Read the pointing data from the file to have the telescope pointings as a property
            for tel_id in self.subarray.tel_ids:
                self._telescope_pointings[tel_id] = read_table(
                    self.input_url,
                    f"{TELESCOPE_CALIBRATION_GROUP}/pointing/tel_{tel_id:03d}",
                )
                # Register the table with the pointing interpolator
                self._pointing_interpolator.add_table(
                    tel_id, self._telescope_pointings[tel_id]
                )

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.file_.close()

    @property
    def is_simulation(self):
        """
        True for files with a simulation group at the root of the file.
        """
        return "simulation" in self.file_.root

    @lazyproperty
    def monitoring_types(self):
        return get_hdf5_monitoring_types(self.file_)

    @lazyproperty
    def has_camera_coefficients(self):
        """
        True for files that contain camera calibration coefficients
        """
        return f"{TELESCOPE_CALIBRATION_GROUP}/camera/coefficients" in self.file_.root

    @lazyproperty
    def has_pixel_statistics(self):
        """
        True for files that contain pixel statistics
        """
        return (
            f"{TELESCOPE_CALIBRATION_GROUP}/camera/pixel_statistics" in self.file_.root
        )

    @lazyproperty
    def has_pointings(self):
        """
        True for files that contain pointing information
        """
        return f"{TELESCOPE_CALIBRATION_GROUP}/pointing" in self.file_.root

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
            event.monitoring.tel[tel_id].camera = self.get_camera_monitoring_container(
                tel_id, event.trigger.time
            )
            # Only overwrite the telescope pointings if explicitly requested
            if self.overwrite_telescope_pointings:
                event.monitoring.tel[
                    tel_id
                ].pointing = self.get_telescope_pointing_container(
                    tel_id, event.trigger.time
                )

    def get_telescope_pointing_container(
        self, tel_id: int, time: astropy.time.Time
    ) -> Generator[TelescopePointingContainer]:
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
        Generator[TelescopePointingContainer]
            A generator yielding the telescope pointing container.
        """

        if self.has_pointings:
            alt, az = self._pointing_interpolator(tel_id, time)
            yield TelescopePointingContainer(altitude=alt, azimuth=az)

    def get_camera_monitoring_container(
        self, tel_id: int, time: astropy.time.Time
    ) -> Generator[MonitoringCameraContainer]:
        """
        Retrieve the camera monitoring container with interpolated or retrieved data for a given time.

        Parameters
        ----------
        tel_id : int
            The telescope ID to retrieve the monitoring data for.
        time : astropy.time.Time
            Target timestamp to find the camera monitoring data for.
        """

        cam_mon_container = MonitoringCameraContainer()
        if self.has_pixel_statistics:
            # Fill the the camera monitoring container with the pixel statistics
            pixel_stats_container = PixelStatisticsContainer()
            for name, interpolator in (
                ("sky_pedestal_image", self._pedestal_image_interpolator),
                ("flatfield_image", self._flatfield_image_interpolator),
                ("flatfield_peak_time", self._flatfield_peak_time_interpolator),
            ):
                # Fill the pixel statistics container
                if self.is_simulation:
                    # In simulations, we only use the first entry of the monitoring data
                    # to fill the pixel statistics, since there is no proper time definition
                    # in simulated observing blocks. Besides, the simulation toolkit is not
                    # varying the observation conditions, e.g. raising pedestal noise level,
                    # within a given simulation run.
                    pixel_stats_container[name] = StatisticsContainer(
                        mean=self._pixel_statistics[tel_id][name]["mean"][0],
                        median=self._pixel_statistics[tel_id][name]["median"][0],
                        std=self._pixel_statistics[tel_id][name]["std"][0],
                        outlier_mask=self._pixel_statistics[tel_id][name][
                            "outlier_mask"
                        ][0],
                    )
                else:
                    # In real data, the statistics are retrieved based on the timestamp of the event
                    # by interpolating the monitoring data with the corresponding ChunkInterpolator
                    stats_data = interpolator(tel_id, time)
                    pixel_stats_container[name] = StatisticsContainer(
                        mean=stats_data["mean"],
                        median=stats_data["median"],
                        std=stats_data["std"],
                        outlier_mask=np.isnan(stats_data["median"]),
                    )
            cam_mon_container["pixel_statistics"] = pixel_stats_container
        if self.has_camera_coefficients:
            # Fill the camera calibration container
            if self.is_simulation:
                # In simulations, we only use the first entry of the monitoring data
                # to fill the telescope calibration coefficients, since there is no
                # proper time definition in simulated observing blocks. Besides, the
                # simulation toolkit is not varying the observation conditions, e.g.
                # raising pedestal noise level, within a given simulation run.
                cam_mon_container["coefficients"] = CameraCalibrationContainer(
                    time=time,
                    pedestal_offset=self._camera_coefficients[tel_id][
                        "pedestal_offset"
                    ][0],
                    factor=self._camera_coefficients[tel_id]["factor"][0],
                    time_shift=self._camera_coefficients[tel_id]["time_shift"][0],
                    outlier_mask=self._camera_coefficients[tel_id]["outlier_mask"][0],
                    is_valid=self._camera_coefficients[tel_id]["is_valid"][0],
                )
            else:
                # In real data, the coefficients are retrieved based on the timestamp of the event.
                # Get the table row corresponding to the target time
                table_row = self._get_table_row(
                    time.to_value("mjd"), self._camera_coefficients_table[tel_id]
                )
                if table_row is None:
                    cam_mon_container["coefficients"] = CameraCalibrationContainer()
                else:
                    cam_mon_container["coefficients"] = CameraCalibrationContainer(
                        time=time,
                        pedestal_offset=table_row["pedestal_offset"],
                        factor=table_row["factor"],
                        time_shift=table_row["time_shift"],
                        outlier_mask=table_row["outlier_mask"],
                        is_valid=table_row["is_valid"],
                    )
        yield cam_mon_container

    def _get_table_row(self, time: float, table: astropy.table.Table):
        """
        Retrieve the row of the table that corresponds to the target time.

        Parameters
        ----------
        time : float
            Target timestamp in MJD to find the interval.
        table : astropy.table.Table
            Table containing ordered timestamp data.

        Returns
        -------
        table_row : astropy.table.Row or None
            The row of the table that corresponds to the target time, or None
            if the target time is not within the lower bound of the table.
        """
        # Find the index of the closest timestamp
        idx = np.searchsorted(table["time"], time, side="right") - 1
        # Check lower bounds when requested timestamp is before the validity start
        if idx < 0:
            raise ValueError(
                f"Out of bounds: Requested timestamp '{time} MJD' is before the "
                f"validity start '{table['time'][0]} MJD' (first entry in the table)."
            )
        # Check if target falls in interval [idx, idx+1) and return the row of the table
        table_row = None
        if table["time"][idx] <= time < table["time"][idx + 1]:
            table_row = table[idx]
        return table_row
