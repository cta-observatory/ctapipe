"""
Handles reading of monitoring files
"""
import logging
from contextlib import ExitStack
from collections.abc import Generator

from ctapipe.src.ctapipe.instrument.optics import FocalLengthKind
import numpy as np
import tables
import astropy
from astropy.table import QTable
from astropy.utils.decorators import lazyproperty


from ..containers import (
    ArrayEventContainer,
    CameraCalibrationContainer,
    MonitoringCameraContainer,
    MonitoringContainer,
    PixelStatisticsContainer,
    StatisticsContainer,
)
from ..core import Provenance
from ..instrument import SubarrayDescription
from .astropy_helpers import read_table
from .metadata import _read_reference_metadata_hdf5
from ..core.component import Component
from ..core.traits import CInt, Path, Set, UseEnum
from ..instrument import SubarrayDescription
from .monitoringtypes import MonitoringTypes
from ..monitoring import (
    FlatfieldImageInterpolator,
    FlatfieldPeakTimeInterpolator,
    PedestalImageInterpolator,
)


__all__ = ["MonitoringSource"]

logger = logging.getLogger(__name__)


def get_hdf5_monitoringtypes(h5file: tables.File | str | Path):
    """Get the monitoring types present in the hdf5 file"""
    monitoring_types = []

    with ExitStack() as stack:
        if not isinstance(h5file, tables.File):
            h5file = stack.enter_context(tables.open_file(h5file))

        if "/dl1/monitoring/telescope/calibration/camera/pixel_statistics" in h5file.root:
            monitoring_types.append(MonitoringTypes.PIXEL_STATISTICS)

        if "/dl1/monitoring/telescope/calibration/camera/coefficients" in h5file.root:
            monitoring_types.append(MonitoringTypes.CAMERA_COEFFICIENTS)

        if "/dl1/monitoring/telescope/pointing/" in h5file.root:
            monitoring_types.append(MonitoringTypes.POINTING)

    return tuple(monitoring_types)


def get_table_row(time: astropy.time.Time, table: astropy.table.Table, time_column='time'):
    """
    Retrieve the row of the table that corresponds to the target time.

    Parameters
    ----------
    time : astropy.time.Time
        Target timestamp to find the interval for.
    table : astropy.table.Table
        Table containing ordered timestamp data.
    time_column : str
        Name of the column containing the timestamps.
        Default is 'time'.
    
    Returns
    -------
    table_row : astropy.table.Row or None
        The row of the table that corresponds to the target time, or None
        if the target time is not within the lower bound of the table.
    """
    # Convert time objects to MJD
    timestamp_mjd = time.to_value("mjd")
    table_times_mjd = table[time_column].to_value("mjd")
    # Find the index of the closest timestamp 
    idx = np.searchsorted(table_times_mjd, timestamp_mjd, side='right') - 1
    # Check lower bounds when requested timestamp is before the validity start
    if idx < 0:
        raise ValueError(
            f"Out of bounds: Requested timestamp '{timestamp_mjd}' is before "
            f"the validity start '{table_times_mjd[0]}' (first entry in the table)."
            )
    # TODO: Out of bounds check for upper limit. >For this, we need the end of the
    # of the obs block? Retrieve it from metadata or directly service data?

    # Check if target falls in interval [idx, idx+1) and return the row of the table
    table_row = None
    if table_times_mjd[idx] <= timestamp_mjd < table_times_mjd[idx + 1]:
        table_row = table[idx]
    return table_row



class MonitoringSource(Component):
    """
    Class for reading monitoring data.

    This class provides a common interface for accessing monitoring data from different monitoring types.
    TODO: Fill proper docstring.

    Parameters
    ----------
    input_url : str | Path
        Path to the input monitoring file.
    allowed_tels: Set or None
        Ids of the telescopes to be included in the data.
        If given, only this subset of telescopes will be present in the
        generated monitoring data. If None, all available telescopes are used.
    """

    input_url = Path(help="Path to the input file containing monitoring data.").tag(config=True)

    allowed_tels = Set(
        trait=CInt(),
        default_value=None,
        allow_none=True,
        help=(
            "list of allowed tel_ids, others will be ignored. "
            "If None, all telescopes in the input stream "
            "will be included"
        ),
    ).tag(config=True)

    # TODO Do we really need this here?
    focal_length_choice = UseEnum(
        FocalLengthKind,
        default_value=FocalLengthKind.EFFECTIVE,
        help=(
            "If both nominal and effective focal lengths are available, "
            " which one to use for the `~ctapipe.coordinates.CameraFrame` attached"
            " to the `~ctapipe.instrument.CameraGeometry` instances in the"
            " `~ctapipe.instrument.SubarrayDescription` which will be used in"
            " CameraFrame to TelescopeFrame coordinate transforms."
            " The 'nominal' focal length is the one used during "
            " the simulation, the 'effective' focal length is computed using specialized "
            " ray-tracing from a point light source"
        ),
    ).tag(config=True)

    def __init__(self, input_url=None, config=None, parent=None, **kwargs):
        """
        MonitoringSource for monitoring files in the standard HDF5 data format

        Parameters:
        -----------
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
        super().__init__(input_url=input_url, config=config, parent=parent, **kwargs)

        self.file_ = tables.open_file(self.input_url)
        meta = _read_reference_metadata_hdf5(self.file_)
        Provenance().add_input_file(
            str(self.input_url), role="Monitoring", reference_meta=meta
        )

        self._full_subarray = SubarrayDescription.from_hdf(
            self.input_url,
            focal_length_choice=self.focal_length_choice,
        )

        if self.allowed_tels:
            self._subarray = self._full_subarray.select_subarray(self.allowed_tels)
        else:
            self._subarray = self._full_subarray

        self._pixel_statistics = {}
        if self.has_pixel_statistics:   
            # Instantiate the chunk interpolators for each table
            self._pedestal_image_interpolator = PedestalImageInterpolator()
            self._flatfield_image_interpolator = FlatfieldImageInterpolator()
            self._flatfield_peak_time_interpolator = FlatfieldPeakTimeInterpolator()
            # Iterate over the telescope IDs
            for tel_id in self._subarray.tel_ids:
                # Process the tables and interpolate the data
                for name, interpolator in (
                    ("sky_pedestal_image", self._pedestal_image_interpolator),
                    ("flatfield_image", self._flatfield_image_interpolator),
                    ("flatfield_peak_time", self._flatfield_peak_time_interpolator),
                ):
                    # Read the tables from the monitoring file requiring all tables to be present
                    self._pixel_statistics[tel_id][name] = read_table(
                        self.input_url,
                        f"/dl1/monitoring/telescope/calibration/camera/pixel_statistics/{name}/tel_{tel_id:03d}",
                    )
                    if not self.is_simulation:
                        # Set outliers to NaNs
                        for col in ["mean", "median", "std"]:
                            self._pixel_statistics[tel_id][name][col][self._pixel_statistics[tel_id][name]["outlier_mask"].data] = np.nan
                        # Register the table with the interpolator
                        interpolator.add_table(tel_id, self._pixel_statistics[tel_id][name])
                        
        if self.has_camera_coefficients:
            self._camera_coefficients = {}
            # Iterate over the telescope IDs and calculate the camera calibration coefficients
            for tel_id in self.subarray.tel_ids:
                # Read the tables from the monitoring file requiring all tables to be present
                self._camera_coefficients[tel_id] = read_table(
                    self.input_url,
                    f"/dl1/monitoring/telescope/calibration/camera/coefficients/tel_{tel_id:03d}",
                )

        # TODO Check how to handle pointing 
        pointing_key = "/configuration/telescope/pointing"
        self._constant_telescope_pointing = {}
        if pointing_key in self.file_.root:
            for h5table in self.file_.root[pointing_key]._f_iter_nodes("Table"):
                tel_id = int(h5table._v_name.partition("tel_")[-1])
                table = QTable(read_table(self.file_, h5table._v_pathname), copy=False)
                table.add_index("obs_id")
                self._constant_telescope_pointing[tel_id] = table
        


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
    def monitoringtypes(self):
        return get_hdf5_monitoringtypes(self.file_)
    
    @lazyproperty
    def has_camera_coefficients(self):
        """
        True for files that contain camera calibration coefficients
        """
        return "/dl1/monitoring/telescope/calibration/camera/coefficients" in self.file_.root

    @lazyproperty
    def has_pixel_statistics(self):
        """
        True for files that contain pixel statistics
        """
        return "/dl1/monitoring/telescope/calibration/camera/pixel_statistics" in self.file_.root

    # TODO Check how to handle pointing 
    @lazyproperty
    def has_pointing(self):
        """
        True for files that contain pointing information
        """
        return "/dl1/monitoring/telescope/pointing" in self.file_.root

    @property
    def subarray(self):
        return self._subarray

    @property
    def camera_coefficients(self):
        return self._camera_coefficients
    
    @property
    def pixel_statistics(self):
        return self._pixel_statistics

    def fill_monitoring_container(self, event: ArrayEventContainer):
        """
        Fill the monitoring container for a given event.

        Parameters
        ----------
        event : ArrayEventContainer
            The event to fill the monitoring container for.
        """
        # Create the monitoring container
        mon_container = MonitoringContainer()
        # Iterate over the telescope IDs
        for tel_id in self._subarray.tel_ids:
            mon_container[tel_id].camera = self.get_camera_monitoring_container(tel_id, event.trigger.time)
        # Fill the monitoring container for the event
        event.monitoring = mon_container
        
    def get_camera_monitoring_container(self, tel_id: int, time: astropy.time.Time) -> Generator[MonitoringCameraContainer]:
        """
        Retrieve the camera monitoring container with interpolated or retrieved data for a given time.

        Parameters
        ----------
        tel_id : int
            Telescope ID.
        time : astropy.time.Time
            Time to interpolate.
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
                        mean = self._pixel_statistics[tel_id][name]["mean"][0],
                        median = self._pixel_statistics[tel_id][name]["median"][0],
                        std = self._pixel_statistics[tel_id][name]["std"][0],
                        outlier_mask = self._pixel_statistics[tel_id][name]["outlier_mask"][0],
                    )
                else:
                    # In real data, the statistics are retrieved based on the timestamp of the event
                    # by interpolating the monitoring data with the corresponding ChunkInterpolator
                    stats_data = interpolator(tel_id, time)
                    pixel_stats_container[name] = StatisticsContainer(
                        mean = stats_data["mean"],
                        median = stats_data["median"],
                        std = stats_data["std"],
                        outlier_mask = np.isnan(stats_data["median"])
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
                    pedestal_offset=self._camera_coefficients[tel_id]["pedestal_offset"][0],
                    factor=self._camera_coefficients[tel_id]["factor"][0],
                    time_shift=self._camera_coefficients[tel_id]["time_shift"][0],
                    outlier_mask=self._camera_coefficients[tel_id]["outlier_mask"][0],
                    is_valid=self._camera_coefficients[tel_id]["is_valid"][0],
                )
            else:
                # In real data, the coefficients are retrieved based on the timestamp of the event
                cam_mon_container["coefficients"] = self.get_camera_calib_coeffs(tel_id, time)
        yield cam_mon_container

    def get_camera_calib_coeffs(self, tel_id: int, time: astropy.time.Time) -> CameraCalibrationContainer:
        """
        Retrieve the camera calibration coefficients for a specific time and fill
        the EventCameraCalibrationContainer with the corresponding values.

        Parameters
        ----------
        tel_id : int
            The telescope ID for which to retrieve the calibration coefficients.
        time : astropy.time.Time
            Target timestamp to find the coefficients for.

        Returns
        -------
        EventCameraCalibrationContainer
            The camera calibration coefficients for the target time, if the
            target time is not within the bounds an empty container is returned.
        """
        # Get the table row corresponding to the target time
        table_row = get_table_row(time, self._camera_coefficients_table[tel_id])
        if table_row is None:
            return CameraCalibrationContainer()
        else:
            return CameraCalibrationContainer(
                time=time,
                pedestal_offset=table_row["pedestal_offset"],
                factor=table_row["factor"],
                time_shift=table_row["time_shift"],
                outlier_mask=table_row["outlier_mask"],
                is_valid=table_row["is_valid"],
            )
