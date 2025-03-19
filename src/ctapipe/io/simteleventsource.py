import enum
import warnings
from contextlib import nullcontext
from enum import Enum, auto, unique
from gzip import GzipFile
from io import BufferedReader
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle, EarthLocation
from astropy.table import Table
from astropy.time import Time

from ..atmosphere import (
    AtmosphereDensityProfile,
    FiveLayerAtmosphereDensityProfile,
    TableAtmosphereDensityProfile,
)
from ..calib.camera.gainselection import GainChannel, GainSelector
from ..compat import COPY_IF_NEEDED
from ..containers import (
    ArrayEventContainer,
    CoordinateFrameType,
    EventIndexContainer,
    EventType,
    ObservationBlockContainer,
    ObservationBlockState,
    ObservingMode,
    PixelStatus,
    PixelStatusContainer,
    PointingContainer,
    PointingMode,
    R0CameraContainer,
    R1CameraContainer,
    SchedulingBlockContainer,
    SchedulingBlockType,
    SimulatedCameraContainer,
    SimulatedEventContainer,
    SimulatedShowerContainer,
    SimulatedShowerDistribution,
    SimulationConfigContainer,
    TelescopeImpactParameterContainer,
    TelescopePointingContainer,
    TelescopeTriggerContainer,
    TriggerContainer,
)
from ..coordinates import CameraFrame, shower_impact_distance
from ..core import Map
from ..core.provenance import Provenance
from ..core.traits import Bool, ComponentName, Float, Integer, Undefined, UseEnum
from ..exceptions import OptionalDependencyMissing
from ..instrument import (
    CameraDescription,
    CameraGeometry,
    CameraReadout,
    FocalLengthKind,
    OpticsDescription,
    ReflectorShape,
    SubarrayDescription,
    TelescopeDescription,
)
from ..instrument.camera import UnknownPixelShapeWarning
from ..instrument.guess import (
    GuessingResult,
    guess_telescope,
    type_from_mirror_area,
    unknown_telescope,
)
from .datalevels import DataLevel
from .eventsource import EventSource

try:
    from eventio import SimTelFile
except ModuleNotFoundError:
    SimTelFile = None


__all__ = [
    "SimTelEventSource",
    "read_atmosphere_profile_from_simtel",
    "AtmosphereProfileKind",
    "MirrorClass",
]

# default for MAX_PHOTOELECTRONS in simtel_array is 2_500_000
# so this should be a safe limit to distinguish "dim image true missing"
# from "extremely bright true image missing", see issue #2344
MISSING_IMAGE_BRIGHTNESS_LIMIT = 1_000_000

# Mapping of SimTelArray Calibration trigger types to EventType:
# from simtelarray: type Dark (0), pedestal (1), in-lid LED (2) or laser/LED (3+) data.
SIMTEL_TO_CTA_EVENT_TYPE = {
    0: EventType.DARK_PEDESTAL,
    1: EventType.SKY_PEDESTAL,
    2: EventType.SINGLE_PE,
    3: EventType.FLATFIELD,
    -1: EventType.OTHER_CALIBRATION,
}

_half_pi = 0.5 * np.pi
_half_pi_maxval = (1 + 1e-6) * _half_pi
_float32_nan = np.float32(np.nan)


def _clip_altitude_if_close(altitude):
    """
    Round absolute values slightly larger than pi/2 in float64 to pi/2

    These can come from simtel_array because float32(pi/2) > float64(pi/2)
    and simtel using float32.

    Astropy complains about these values, so we fix them here.
    """
    if altitude > _half_pi and altitude < _half_pi_maxval:
        return _half_pi

    if altitude < -_half_pi and altitude > -_half_pi_maxval:
        return -_half_pi

    return altitude


@enum.unique
class MirrorClass(enum.Enum):
    """Enum for the sim_telarray MIRROR_CLASS values"""

    SINGLE_SEGMENTED_MIRROR = 0
    SINGLE_SOLID_PARABOLIC_MIRROR = 1
    DUAL_MIRROR = 2


GRAMMAGE_UNIT = u.g / (u.cm**2)
NANOSECONDS_PER_DAY = (1 * u.day).to_value(u.ns)


def parse_simtel_time(simtel_time):
    """Convert a unix time second / nanosecond tuple into astropy.time.Time"""
    return Time(
        simtel_time[0],
        simtel_time[1] * 1e-9,
        format="unix",
        scale="utc",  # ns to s
    )


def _location_from_meta(global_meta):
    """Extract reference location of subarray from metadata."""
    lat = global_meta.get(b"*LATITUDE")
    lon = global_meta.get(b"*LONGITUDE")
    height = global_meta.get(b"ALTITUDE")

    if lat is None or lon is None or height is None:
        return None

    return EarthLocation(
        lon=float(lon) * u.deg,
        lat=float(lat) * u.deg,
        height=float(height) * u.m,
    )


def build_camera(simtel_telescope, telescope, frame):
    """Create CameraDescription from eventio data structures"""
    camera_settings = simtel_telescope["camera_settings"]
    pixel_settings = simtel_telescope["pixel_settings"]
    camera_organization = simtel_telescope["camera_organization"]

    pixel_shape = camera_settings["pixel_shape"][0]
    try:
        pix_type, pix_rotation = CameraGeometry.simtel_shape_to_type(pixel_shape)
    except ValueError:
        warnings.warn(
            f"Unknown pixel_shape {pixel_shape} for camera_type {telescope.camera_name}",
            UnknownPixelShapeWarning,
        )
        pix_type = "hexagon"
        pix_rotation = "0d"

    geometry = CameraGeometry(
        telescope.camera_name,
        pix_id=np.arange(camera_settings["n_pixels"]),
        pix_x=u.Quantity(camera_settings["pixel_x"], u.m),
        pix_y=u.Quantity(camera_settings["pixel_y"], u.m),
        pix_area=u.Quantity(camera_settings["pixel_area"], u.m**2),
        pix_type=pix_type,
        pix_rotation=pix_rotation,
        cam_rotation=-Angle(camera_settings["cam_rot"], u.rad),
        apply_derotation=True,
        frame=frame,
    )
    readout = CameraReadout(
        telescope.camera_name,
        sampling_rate=u.Quantity(
            1.0 / pixel_settings["time_slice"].astype(np.float64), u.GHz
        ),
        reference_pulse_shape=pixel_settings["ref_shape"].astype(
            "float64", copy=COPY_IF_NEEDED
        ),
        reference_pulse_sample_width=u.Quantity(
            pixel_settings["ref_step"], u.ns, dtype="float64"
        ),
        n_channels=camera_organization["n_gains"],
        n_pixels=camera_organization["n_pixels"],
        n_samples=pixel_settings["sum_bins"],
    )

    return CameraDescription(
        name=telescope.camera_name,
        geometry=geometry,
        readout=readout,
    )


def _telescope_from_meta(telescope_meta, mirror_area):
    optics_name = telescope_meta.get(b"OPTICS_CONFIG_NAME")
    camera_name = telescope_meta.get(b"CAMERA_CONFIG_NAME")
    mirror_class = telescope_meta.get(b"MIRROR_CLASS")

    if optics_name is None or camera_name is None or mirror_class is None:
        return None

    telescope_type = type_from_mirror_area(mirror_area)

    mirror_class = MirrorClass(int(mirror_class))

    reflector_shape = ReflectorShape.UNKNOWN
    if mirror_class is MirrorClass.DUAL_MIRROR:
        reflector_shape = ReflectorShape.SCHWARZSCHILD_COUDER
    elif mirror_class is MirrorClass.SINGLE_SEGMENTED_MIRROR:
        if int(telescope_meta.get(b"PARABOLIC_DISH", 0)) == 1:
            reflector_shape = ReflectorShape.PARABOLIC
        else:
            reflector_shape = ReflectorShape.DAVIES_COTTON

            shape_length = float(telescope_meta.get(b"DISH_SHAPE_Length", 0))
            focal_length = float(telescope_meta.get(b"FOCAL_Length", 0))

            if shape_length != focal_length:
                reflector_shape = ReflectorShape.HYBRID

    n_mirrors = 2 if mirror_class is MirrorClass.DUAL_MIRROR else 1

    return GuessingResult(
        type=telescope_type,
        reflector_shape=reflector_shape,
        name=optics_name.decode("utf-8"),
        camera_name=camera_name.decode("utf-8"),
        n_mirrors=n_mirrors,
    )


def apply_simtel_r1_calibration(
    r0_waveforms, pedestal, dc_to_pe, gain_selector, calib_scale=1.0, calib_shift=0.0
):
    """
    Perform the R1 calibration for R0 simtel waveforms. This includes:
        - Gain selection
        - Pedestal subtraction
        - Conversion of samples into units proportional to photoelectrons
          (If the full signal in the waveform was integrated, then the resulting
          value would be in photoelectrons.)
          (Also applies flat-fielding)

    Parameters
    ----------
    r0_waveforms : ndarray
        Raw ADC waveforms from a simtel file. All gain channels available.
        Shape: (n_channels, n_pixels, n_samples)
    pedestal : ndarray
        Pedestal stored in the simtel file for each gain channel
        Shape: (n_channels, n_pixels)
    dc_to_pe : ndarray
        Conversion factor between R0 waveform samples and ~p.e., stored in the
        simtel file for each gain channel
        Shape: (n_channels, n_pixels)
    gain_selector : ctapipe.calib.camera.gainselection.GainSelector
    calib_scale : float
        Extra global scale factor for calibration.
        Conversion factor to transform the integrated charges
        (in ADC counts) into number of photoelectrons on top of dc_to_pe.
        Defaults to no scaling.
    calib_shift: float
        Shift the resulting R1 output in p.e. for simulating miscalibration.
        Defaults to no shift.

    Returns
    -------
    r1_waveforms : ndarray
        Calibrated waveforms
        Shape: (n_channels, n_pixels, n_samples)
    selected_gain_channel : ndarray
        The gain channel selected for each pixel
        Shape: (n_pixels)
    """
    n_pixels = r0_waveforms.shape[-2]
    ped = pedestal[..., np.newaxis]
    DC_to_PHE = dc_to_pe[..., np.newaxis]
    gain = DC_to_PHE * calib_scale

    r1_waveforms = (r0_waveforms - ped) * gain + calib_shift

    if gain_selector is not None:
        selected_gain_channel = gain_selector(r0_waveforms)
        r1_waveforms = r1_waveforms[
            np.newaxis, selected_gain_channel, np.arange(n_pixels)
        ]
    else:
        selected_gain_channel = None

    return r1_waveforms, selected_gain_channel


@unique
class AtmosphereProfileKind(Enum):
    """
    choice for which kind of atmosphere density profile to load if more than
    one is present"""

    #: Don't load a profile
    NONE = auto()

    #: Use a TableAtmosphereDensityProfile
    TABLE = auto()

    #: Use a FiveLayerAtmosphereDensityProfile
    FIVELAYER = auto()

    #: Try TABLE first, and if it doesn't exist, use FIVELAYER
    AUTO = auto()


def read_atmosphere_profile_from_simtel(
    simtelfile: str | Path | SimTelFile, kind=AtmosphereProfileKind.AUTO
) -> TableAtmosphereDensityProfile | None:
    """Read an atmosphere profile from a SimTelArray file as an astropy Table

    Parameters
    ----------
    simtelfile: str | SimTelFile
        filename of a SimTelArray file containing an atmosphere profile
    kind: AtmosphereProfileKind | str
        which type of model to load. In AUTO mode, table is tried first,
        and if it doesn't exist, fivelayer is used.

    Returns
    -------
    Optional[TableAtmosphereDensityProfile]:
        Profile read from a table, with interpolation

    """
    if not isinstance(kind, AtmosphereProfileKind):
        raise TypeError(
            "read_atmosphere_profile_from_simtel: kind should be a AtmosphereProfileKind"
        )

    profiles = []

    if kind == AtmosphereProfileKind.NONE:
        return None

    if isinstance(simtelfile, str | Path):
        context_manager = SimTelFile(simtelfile)
        # FIXME: simtel files currently do not have CTAO reference
        # metadata, should be set to True once we store metadata
        Provenance().add_input_file(
            filename=simtelfile,
            role="ctapipe.atmosphere.AtmosphereDensityProfile",
            add_meta=False,
        )

    else:
        context_manager = nullcontext(simtelfile)

    with context_manager as simtel:
        if (
            not hasattr(simtel, "atmospheric_profiles")
            or len(simtel.atmospheric_profiles) == 0
        ):
            return []

        for atmo in simtel.atmospheric_profiles:
            metadata = dict(
                observation_level=atmo["obslevel"] * u.cm,
                atmosphere_id=atmo["id"],
                atmosphere_name=atmo["name"].decode("utf-8"),
                atmosphere_height=atmo["htoa"] * u.cm,
            )

            if "altitude_km" in atmo and kind in {
                AtmosphereProfileKind.TABLE,
                AtmosphereProfileKind.AUTO,
            }:
                table = Table(
                    dict(
                        height=atmo["altitude_km"] * u.km,
                        density=atmo["rho"] * u.g / u.cm**3,
                        column_density=atmo["thickness"] * u.g / u.cm**2,
                    ),
                    meta=metadata,
                )
                profiles.append(TableAtmosphereDensityProfile(table=table))

            elif (
                kind == AtmosphereProfileKind.FIVELAYER
                and "five_layer_atmosphere" in atmo
            ):
                profiles.append(
                    FiveLayerAtmosphereDensityProfile.from_array(
                        atmo["five_layer_atmosphere"], metadata=metadata
                    )
                )

            else:
                raise ValueError(f"Couldn't  load requested profile '{kind}'")

    if profiles:
        return profiles[0]

    return None


class SimTelEventSource(EventSource):
    """
    Read events from a SimTelArray data file (in EventIO format).

    ctapipe makes use of the sim_telarray metadata system to fill some
    information not directly accessible from the data inside the files itself.
    Make sure you set this parameters in the simulation configuration to fully
    make use of ctapipe. In future, ctapipe might require these metadata parameters.

    This includes:

    * Reference Point of the telescope coordinates. Make sure to include the
        user-defined parameters ``LONGITUDE`` and ``LATITUDE`` with the geodetic
        coordinates of the array reference point. Also make sure the ``ALTITUDE``
        config parameter is included in the global metadata.

    * Names of the optical structures and the cameras are read from
        ``OPTICS_CONFIG_NAME`` and ``CAMERA_CONFIG_NAME``, make sure to include
        these in the telescope meta.

    * The ``MIRROR_CLASS`` should also be included in the telescope meta
        to correctly setup coordinate transforms.

    If these parameters are not included in the input data, ctapipe will
    fallback guesses these based on available data and the list of known telescopes
    for `ctapipe.instrument.guess_telescope`.
    """

    skip_calibration_events = Bool(True, help="Skip calibration events").tag(
        config=True
    )
    back_seekable = Bool(
        False,
        help=(
            "Require the event source to be backwards seekable."
            " This will reduce in slower read speed for gzipped files"
            " and is not possible for zstd compressed files"
        ),
    ).tag(config=True)

    focal_length_choice = UseEnum(
        FocalLengthKind,
        default_value=FocalLengthKind.EFFECTIVE,
        help=(
            "If both nominal and effective focal lengths are available in the"
            " SimTelArray file, which one to use for the `~ctapipe.coordinates.CameraFrame`"
            " attached to the `~ctapipe.instrument.CameraGeometry` instances in"
            " the `~ctapipe.instrument.SubarrayDescription`, which will be used in"
            " CameraFrame to TelescopeFrame coordinate transforms. "
            " The 'nominal' focal length is the one used during "
            " the simulation, the 'effective' focal length is computed using specialized "
            " ray-tracing from a point light source"
        ),
    ).tag(config=True)

    gain_selector_type = ComponentName(
        GainSelector, default_value="ThresholdGainSelector"
    ).tag(config=True)

    calib_scale = Float(
        default_value=1.0,
        help=(
            "Factor to transform ADC counts into number of photoelectrons."
            " Corrects the DC_to_PHE factor."
        ),
    ).tag(config=True)

    calib_shift = Float(
        default_value=0.0,
        help=(
            "Factor to shift the R1 photoelectron samples. "
            "Can be used to simulate mis-calibration."
        ),
    ).tag(config=True)

    atmosphere_profile_choice = UseEnum(
        AtmosphereProfileKind,
        default_value=AtmosphereProfileKind.AUTO,
        help=(
            "Which type of atmosphere density profile to load "
            "from the file, in case more than one exists.  If set "
            "to AUTO, TABLE will be attempted first and if missing, "
            "FIVELAYER will be loaded."
        ),
    ).tag(config=True)

    override_obs_id = Integer(
        default_value=None,
        allow_none=True,
        help="Use the given obs_id instead of the run number from sim_telarray",
    ).tag(config=True)

    select_gain = Bool(
        default_value=None,
        allow_none=True,
        help=(
            "Whether to perform gain selection. The default (None) means only"
            " select gain of physics events, not of calibration events."
        ),
    ).tag(config=True)

    def __init__(self, input_url=Undefined, config=None, parent=None, **kwargs):
        """
        EventSource for simtelarray files using the pyeventio library.

        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool : ctapipe.core.Tool
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        gain_selector : ctapipe.calib.camera.gainselection.GainSelector
            The GainSelector to use. If None, then ThresholdGainSelector will be used.
        kwargs
        """
        super().__init__(input_url=input_url, config=config, parent=parent, **kwargs)
        if SimTelFile is None:
            raise OptionalDependencyMissing("eventio")

        self.file_ = SimTelFile(
            self.input_url.expanduser(),
            allowed_telescopes=self.allowed_tels,
            skip_calibration=self.skip_calibration_events,
            zcat=not self.back_seekable,
        )
        # TODO: read metadata from simtel metaparams once we have files that
        # actually provide the reference metadata.
        Provenance().add_input_file(
            str(self.input_url), role="DL0/Event", add_meta=False
        )

        if self.back_seekable and self.is_stream:
            raise OSError("back seekable was required but not possible for inputfile")
        (
            self._scheduling_blocks,
            self._observation_blocks,
        ) = self._fill_scheduling_and_observation_blocks()
        self._simulation_config = self._parse_simulation_header()
        self._subarray_info = self.prepare_subarray_info(
            self.file_.telescope_descriptions, self.file_.header
        )
        self.start_pos = self.file_.tell()

        self.gain_selector = GainSelector.from_name(
            self.gain_selector_type, parent=self
        )

        self._atmosphere_density_profile = read_atmosphere_profile_from_simtel(
            self.file_, kind=self.atmosphere_profile_choice
        )

        self._has_true_image = None
        self.log.debug(f"Using gain selector {self.gain_selector}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.file_.close()

    @property
    def is_simulation(self):
        return True

    @property
    def datalevels(self):
        return (DataLevel.R0, DataLevel.R1)

    @property
    def simulation_config(self) -> dict[int, SimulationConfigContainer]:
        return self._simulation_config

    @property
    def atmosphere_density_profile(self) -> AtmosphereDensityProfile:
        return self._atmosphere_density_profile

    @property
    def observation_blocks(self) -> dict[int, ObservationBlockContainer]:
        """
        Obtain the ObservationConfigurations from the EventSource, indexed by obs_id
        """
        return self._observation_blocks

    @property
    def scheduling_blocks(self) -> dict[int, SchedulingBlockContainer]:
        """
        Obtain the ObservationConfigurations from the EventSource, indexed by obs_id
        """
        return self._scheduling_blocks

    @property
    def is_stream(self):
        return not isinstance(self.file_._filehandle, BufferedReader | GzipFile)

    def prepare_subarray_info(self, telescope_descriptions, header):
        """
        Constructs a SubarrayDescription object from the
        ``telescope_descriptions`` given by ``SimTelFile``

        Parameters
        ----------
        telescope_descriptions: dict
            telescope descriptions as given by ``SimTelFile.telescope_descriptions``
        header: dict
            header as returned by ``SimTelFile.header``

        Returns
        -------
        SubarrayDescription :
            instrumental information
        """

        tel_descriptions = {}  # tel_id : TelescopeDescription
        tel_positions = {}  # tel_id : TelescopeDescription

        self.telescope_indices_original = {}

        for tel_id, telescope_description in telescope_descriptions.items():
            cam_settings = telescope_description["camera_settings"]

            n_pixels = cam_settings["n_pixels"]
            mirror_area = u.Quantity(cam_settings["mirror_area"], u.m**2)

            equivalent_focal_length = u.Quantity(cam_settings["focal_length"], u.m)
            effective_focal_length = u.Quantity(
                cam_settings.get("effective_focal_length", np.nan), u.m
            )

            telescope = _telescope_from_meta(
                self.file_.telescope_meta.get(tel_id, {}),
                mirror_area,
            )

            if telescope is None:
                try:
                    telescope = guess_telescope(
                        n_pixels,
                        equivalent_focal_length,
                        cam_settings["n_mirrors"],
                    )
                except ValueError:
                    telescope = unknown_telescope(mirror_area, n_pixels)

                # TODO: switch to warning or even an exception once
                # we start relying on this.
                self.log.debug(
                    "Could not determine telescope from sim_telarray metadata,"
                    " guessing using builtin lookup-table: %d: %s",
                    tel_id,
                    telescope,
                )

            optics = OpticsDescription(
                name=telescope.name,
                size_type=telescope.type,
                reflector_shape=telescope.reflector_shape,
                n_mirrors=telescope.n_mirrors,
                equivalent_focal_length=equivalent_focal_length,
                effective_focal_length=effective_focal_length,
                mirror_area=mirror_area,
                n_mirror_tiles=cam_settings["n_mirrors"],
            )

            if self.focal_length_choice is FocalLengthKind.EFFECTIVE:
                if np.isnan(effective_focal_length):
                    raise RuntimeError(
                        "`SimTelEventSource.focal_length_choice` was set to 'EFFECTIVE'"
                        ", but the effective focal length was not present in the file."
                        " Set `focal_length_choice='EQUIVALENT'` or make sure"
                        " input files contain the effective focal length"
                    )
                focal_length = effective_focal_length
            elif self.focal_length_choice is FocalLengthKind.EQUIVALENT:
                focal_length = equivalent_focal_length
            else:
                raise ValueError(
                    f"Invalid focal length choice: {self.focal_length_choice}"
                )

            camera = build_camera(
                telescope_description,
                telescope,
                frame=CameraFrame(focal_length=focal_length),
            )

            tel_descriptions[tel_id] = TelescopeDescription(
                name=telescope.name,
                optics=optics,
                camera=camera,
            )

            tel_idx = np.where(header["tel_id"] == tel_id)[0][0]
            self.telescope_indices_original[tel_id] = tel_idx
            tel_positions[tel_id] = header["tel_pos"][tel_idx] * u.m

        name = self.file_.global_meta.get(b"ARRAY_CONFIG_NAME", b"MonteCarloArray")

        reference_location = _location_from_meta(self.file_.global_meta)
        if reference_location is None:
            reference_location = self._make_dummy_location()

        subarray = SubarrayDescription(
            name=name.decode(),
            tel_positions=tel_positions,
            tel_descriptions=tel_descriptions,
            reference_location=reference_location,
        )

        self.n_telescopes_original = len(subarray)

        if self.allowed_tels:
            subarray = subarray.select_subarray(self.allowed_tels)

        return subarray

    def _make_dummy_location(self):
        """
        Returns
        -------
        EarthLocation:
            Dummy earth location that is at the correct height (as given in the
            SimulationConfigContainer), but with the lat/lon set to (0,0), i.e.
            on "Null Island"
        """
        return EarthLocation(
            lon=0 * u.deg,
            lat=0 * u.deg,
            height=self._simulation_config[self.obs_id].prod_site_alt,
        )

    @staticmethod
    def is_compatible(file_path):
        try:
            from eventio.file_types import is_eventio
        except ModuleNotFoundError:
            raise OptionalDependencyMissing("eventio") from None

        path = Path(file_path).expanduser()
        if not path.is_file():
            return False
        return is_eventio(path)

    @property
    def subarray(self):
        return self._subarray_info

    @property
    def simulated_shower_distributions(self) -> dict[int, SimulatedShowerDistribution]:
        if self.file_.histograms is None:
            warnings.warn(
                "eventio file has no histograms (yet)."
                " Histograms are stored at the end of the file and can only be read after all events have been consumed."
                " Setting max-events will also prevent the histograms from being available."
            )
            return {}

        shower_hist = next(hist for hist in self.file_.histograms if hist["id"] == 6)

        xbins = np.linspace(
            shower_hist["lower_x"],
            shower_hist["upper_x"],
            shower_hist["n_bins_x"] + 1,
        )
        ybins = np.linspace(
            shower_hist["lower_y"],
            shower_hist["upper_y"],
            shower_hist["n_bins_y"] + 1,
        )

        container = SimulatedShowerDistribution(
            obs_id=self.obs_id,
            hist_id=shower_hist["id"],
            n_entries=shower_hist["entries"],
            bins_core_dist=u.Quantity(xbins, u.m, copy=False),
            bins_energy=u.Quantity(10**ybins, u.TeV, copy=False),
            histogram=shower_hist["data"],
        )

        return {self.obs_id: container}

    def _generator(self):
        if self.file_.tell() > self.start_pos:
            self.file_._next_header_pos = 0
            warnings.warn("Backseeking to start of file.")

        try:
            yield from self._generate_events()
        except EOFError:
            msg = 'EOFError reading from "{input_url}". Might be truncated'.format(
                input_url=self.input_url
            )
            self.log.warning(msg)
            warnings.warn(msg)

    def _generate_events(self):
        # for events without event_id, we use negative event_ids
        pseudo_event_id = 0

        for counter, array_event in enumerate(self.file_):
            event_id = array_event.get("event_id", 0)
            if event_id == 0:
                pseudo_event_id -= 1
                event_id = pseudo_event_id

            obs_id = self.obs_id

            trigger = self._fill_trigger_info(array_event)
            if trigger.event_type == EventType.SUBARRAY:
                shower = self._fill_simulated_event_information(array_event)
            else:
                shower = None

            data = ArrayEventContainer(
                simulation=SimulatedEventContainer(shower=shower),
                pointing=self._fill_array_pointing(),
                index=EventIndexContainer(obs_id=obs_id, event_id=event_id),
                count=counter,
                trigger=trigger,
            )
            data.meta["origin"] = "hessio"
            data.meta["input_url"] = self.input_url
            data.meta["max_events"] = self.max_events

            telescope_events = array_event["telescope_events"]
            tracking_positions = array_event["tracking_positions"]

            photoelectron_sums = array_event.get("photoelectron_sums")
            if photoelectron_sums is not None:
                true_image_sums = photoelectron_sums.get(
                    "n_pe", np.full(self.n_telescopes_original, np.nan)
                )
            else:
                true_image_sums = np.full(self.n_telescopes_original, np.nan)

            if data.simulation.shower is not None:
                # compute impact distances of the shower to the telescopes
                impact_distances = shower_impact_distance(
                    shower_geom=data.simulation.shower, subarray=self.subarray
                )
            else:
                impact_distances = np.full(len(self.subarray), np.nan) * u.m

            for tel_id, telescope_event in telescope_events.items():
                if tel_id not in trigger.tels_with_trigger:
                    # skip additional telescopes that have data but did not
                    # participate in the subarray trigger decision for this stereo event
                    # see #2660 for details
                    self.log.warning(
                        "Encountered telescope event not present in"
                        " stereo trigger information, skipping."
                        " event_id = %d, tel_id = %d, tels_with_trigger: %s",
                        event_id,
                        tel_id,
                        trigger.tels_with_trigger,
                    )
                    continue

                adc_samples = telescope_event.get("adc_samples")
                if adc_samples is None:
                    adc_samples = telescope_event["adc_sums"][:, :, np.newaxis]

                n_gains, n_pixels, n_samples = adc_samples.shape
                true_image = (
                    array_event.get("photoelectrons", {})
                    .get(tel_id - 1, {})
                    .get("photoelectrons", None)
                )
                true_image_sum = true_image_sums[
                    self.telescope_indices_original[tel_id]
                ]

                if self._has_true_image is None:
                    self._has_true_image = true_image is not None

                if self._has_true_image and true_image is None:
                    if true_image_sum > MISSING_IMAGE_BRIGHTNESS_LIMIT:
                        self.log.warning(
                            "Encountered extremely bright telescope event with missing true_image in"
                            "file that has true images: event_id = %d, tel_id = %d."
                            "event might be truncated, skipping telescope event",
                            event_id,
                            tel_id,
                        )
                        continue
                    else:
                        self.log.info(
                            "telescope event event_id = %d, tel_id = %d is missing true image",
                            event_id,
                            tel_id,
                        )
                        true_image = np.full(n_pixels, -1, dtype=np.int32)

                if data.simulation is not None:
                    if data.simulation.shower is not None:
                        impact_container = TelescopeImpactParameterContainer(
                            distance=impact_distances[
                                self.subarray.tel_index_array[tel_id]
                            ],
                            distance_uncert=0 * u.m,
                            prefix="true_impact",
                        )
                    else:
                        impact_container = TelescopeImpactParameterContainer(
                            prefix="true_impact",
                        )

                    data.simulation.tel[tel_id] = SimulatedCameraContainer(
                        true_image_sum=true_image_sum,
                        true_image=true_image,
                        impact=impact_container,
                    )

                data.pointing.tel[tel_id] = self._fill_event_pointing(
                    tracking_positions[tel_id]
                )

                data.r0.tel[tel_id] = R0CameraContainer(waveform=adc_samples)

                cam_mon = array_event["camera_monitorings"][tel_id]
                pedestal = cam_mon["pedestal"] / cam_mon["n_ped_slices"]
                dc_to_pe = array_event["laser_calibrations"][tel_id]["calib"]

                # fill dc_to_pe and pedestal_per_sample info into monitoring
                # container
                mon = data.mon.tel[tel_id]
                mon.calibration.dc_to_pe = dc_to_pe
                mon.calibration.pedestal_per_sample = pedestal
                mon.pixel_status = self._fill_mon_pixels_status(tel_id)

                select_gain = self.select_gain is True or (
                    self.select_gain is None
                    and trigger.event_type is EventType.SUBARRAY
                )
                if select_gain:
                    gain_selector = self.gain_selector
                else:
                    gain_selector = None

                r1_waveform, selected_gain_channel = apply_simtel_r1_calibration(
                    adc_samples,
                    pedestal,
                    dc_to_pe,
                    gain_selector,
                    self.calib_scale,
                    self.calib_shift,
                )

                pixel_status = self._get_r1_pixel_status(
                    tel_id=tel_id,
                    selected_gain_channel=selected_gain_channel,
                )
                data.r1.tel[tel_id] = R1CameraContainer(
                    event_type=trigger.event_type,
                    waveform=r1_waveform,
                    selected_gain_channel=selected_gain_channel,
                    pixel_status=pixel_status,
                )

                # get time_shift from laser calibration
                time_calib = array_event["laser_calibrations"][tel_id]["tm_calib"]
                dl1_calib = data.calibration.tel[tel_id].dl1
                dl1_calib.time_shift = time_calib

            yield data

    def _get_r1_pixel_status(self, tel_id, selected_gain_channel):
        tel_desc = self.file_.telescope_descriptions[tel_id]
        n_pixels = tel_desc["camera_organization"]["n_pixels"]
        pixel_status = np.zeros(n_pixels, dtype=np.uint8)

        high_gain_stored = selected_gain_channel == GainChannel.HIGH
        low_gain_stored = selected_gain_channel == GainChannel.LOW

        # set gain bits
        pixel_status[high_gain_stored] |= np.uint8(PixelStatus.HIGH_GAIN_STORED)
        pixel_status[low_gain_stored] |= np.uint8(PixelStatus.LOW_GAIN_STORED)

        # reset gain bits for completely disabled pixels
        disabled = tel_desc["disabled_pixels"]["HV_disabled"]
        channel_bits = PixelStatus.HIGH_GAIN_STORED | PixelStatus.LOW_GAIN_STORED
        pixel_status[disabled] &= ~np.uint8(channel_bits)

        return pixel_status

    def _fill_mon_pixels_status(self, tel_id):
        tel = self.file_.telescope_descriptions[tel_id]
        n_pixels = tel["camera_organization"]["n_pixels"]
        n_gains = tel["camera_organization"]["n_gains"]

        disabled_ids = tel["disabled_pixels"]["HV_disabled"]
        disabled_pixels = np.zeros((n_gains, n_pixels), dtype=bool)
        disabled_pixels[:, disabled_ids] = True

        return PixelStatusContainer(
            hardware_failing_pixels=disabled_pixels,
            pedestal_failing_pixels=disabled_pixels.copy(),
            flatfield_failing_pixels=disabled_pixels.copy(),
        )

    @staticmethod
    def _fill_event_pointing(tracking_position):
        azimuth_raw = tracking_position["azimuth_raw"]
        altitude_raw = tracking_position["altitude_raw"]
        azimuth_cor = tracking_position.get("azimuth_cor", np.nan)
        altitude_cor = tracking_position.get("altitude_cor", np.nan)

        # take pointing corrected position if available
        if np.isnan(azimuth_cor):
            azimuth = u.Quantity(azimuth_raw, u.rad, copy=COPY_IF_NEEDED)
        else:
            azimuth = u.Quantity(azimuth_cor, u.rad, copy=COPY_IF_NEEDED)

        # take pointing corrected position if available
        if np.isnan(altitude_cor):
            altitude = u.Quantity(
                _clip_altitude_if_close(altitude_raw), u.rad, copy=COPY_IF_NEEDED
            )
        else:
            altitude = u.Quantity(
                _clip_altitude_if_close(altitude_cor), u.rad, copy=COPY_IF_NEEDED
            )

        return TelescopePointingContainer(azimuth=azimuth, altitude=altitude)

    def _fill_trigger_info(self, array_event):
        trigger = array_event["trigger_information"]

        if array_event["type"] == "data":
            event_type = EventType.SUBARRAY

        elif array_event["type"] == "calibration":
            # if using eventio >= 1.1.1, we can use the calibration_type
            event_type = SIMTEL_TO_CTA_EVENT_TYPE.get(
                array_event.get("calibration_type", -1), EventType.OTHER_CALIBRATION
            )

        else:
            event_type = EventType.UNKNOWN

        if self.allowed_tels:
            tels_with_trigger = np.intersect1d(
                trigger["triggered_telescopes"],
                self.subarray.tel_ids,
                assume_unique=True,
            )
        else:
            tels_with_trigger = trigger["triggered_telescopes"]

        central_time = parse_simtel_time(trigger["gps_time"])

        tel = Map(TelescopeTriggerContainer)
        for tel_id, time in zip(
            trigger["triggered_telescopes"], trigger["trigger_times"]
        ):
            if self.allowed_tels and tel_id not in self.allowed_tels:
                continue

            # telescope time is relative to central trigger in ns
            time = Time(
                central_time.jd1,
                central_time.jd2 + time / NANOSECONDS_PER_DAY,
                scale=central_time.scale,
                format="jd",
            )

            # triggered pixel info
            n_trigger_pixels = -1
            trigger_pixels = None

            tel_event = array_event["telescope_events"].get(tel_id)
            if tel_event:
                # code 0 = trigger pixels
                pixel_list = tel_event["pixel_lists"].get(0)
                if pixel_list:
                    n_trigger_pixels = pixel_list["pixels"]
                    trigger_pixels = pixel_list["pixel_list"]

            tel[tel_id] = TelescopeTriggerContainer(
                time=time,
                n_trigger_pixels=n_trigger_pixels,
                trigger_pixels=trigger_pixels,
            )
        return TriggerContainer(
            event_type=event_type,
            time=central_time,
            tels_with_trigger=tels_with_trigger,
            tel=tel,
        )

    def _fill_array_pointing(self):
        if self.file_.header["tracking_mode"] == 0:
            az, alt = self.file_.header["direction"]
            return PointingContainer(
                array_altitude=u.Quantity(alt, u.rad),
                array_azimuth=u.Quantity(az, u.rad),
            )
        else:
            ra, dec = self.file_.header["direction"]
            return PointingContainer(
                array_ra=u.Quantity(ra, u.rad),
                array_dec=u.Quantity(dec, u.rad),
            )

    def _parse_simulation_header(self):
        """
        Parse the simulation infos and return a dict with
        observation ids mapped to SimulationConfigContainers.
        As merged simtel files are not supported at this
        point in time, this dictionary will always have
        length 1.
        """
        obs_id = self.obs_id
        # With only one run, we can take the first entry:
        mc_run_head = self.file_.mc_run_headers[-1]

        simulation_config = SimulationConfigContainer(
            run_number=self.file_.header["run"],
            detector_prog_start=mc_run_head["detector_prog_start"],
            detector_prog_id=mc_run_head["detector_prog_id"],
            corsika_version=mc_run_head["shower_prog_vers"],
            simtel_version=mc_run_head["detector_prog_vers"],
            energy_range_min=mc_run_head["E_range"][0] * u.TeV,
            energy_range_max=mc_run_head["E_range"][1] * u.TeV,
            prod_site_B_total=mc_run_head["B_total"] * u.uT,
            prod_site_B_declination=Angle(mc_run_head["B_declination"], u.rad),
            prod_site_B_inclination=Angle(mc_run_head["B_inclination"], u.rad),
            prod_site_alt=mc_run_head["obsheight"] * u.m,
            spectral_index=mc_run_head["spectral_index"],
            shower_prog_start=mc_run_head["shower_prog_start"],
            shower_prog_id=mc_run_head["shower_prog_id"],
            n_showers=mc_run_head["n_showers"],
            shower_reuse=mc_run_head["n_use"],
            max_alt=_clip_altitude_if_close(mc_run_head["alt_range"][1]) * u.rad,
            min_alt=_clip_altitude_if_close(mc_run_head["alt_range"][0]) * u.rad,
            max_az=mc_run_head["az_range"][1] * u.rad,
            min_az=mc_run_head["az_range"][0] * u.rad,
            diffuse=mc_run_head["diffuse"],
            max_viewcone_radius=mc_run_head["viewcone"][1] * u.deg,
            min_viewcone_radius=mc_run_head["viewcone"][0] * u.deg,
            max_scatter_range=mc_run_head["core_range"][1] * u.m,
            min_scatter_range=mc_run_head["core_range"][0] * u.m,
            core_pos_mode=mc_run_head["core_pos_mode"],
            atmosphere=mc_run_head["atmosphere"],
            corsika_iact_options=mc_run_head["corsika_iact_options"],
            corsika_low_E_model=mc_run_head["corsika_low_E_model"],
            corsika_high_E_model=mc_run_head["corsika_high_E_model"],
            corsika_bunchsize=mc_run_head["corsika_bunchsize"],
            corsika_wlen_min=mc_run_head["corsika_wlen_min"] * u.nm,
            corsika_wlen_max=mc_run_head["corsika_wlen_max"] * u.nm,
            corsika_low_E_detail=mc_run_head["corsika_low_E_detail"],
            corsika_high_E_detail=mc_run_head["corsika_high_E_detail"],
        )
        return {obs_id: simulation_config}

    def _fill_scheduling_and_observation_blocks(self):
        """fill scheduling and observation blocks must be run after the
        simulation config is filled
        """

        az, alt = self.file_.header["direction"]

        # this event source always contains only a single OB, so we can
        # also assign a single obs_id
        if self.override_obs_id is not None:
            self.obs_id = self.override_obs_id
        else:
            self.obs_id = self.file_.header["run"]

        # simulations at the moment do not have SBs, use OB id
        self.sb_id = self.obs_id

        sb_dict = {
            self.sb_id: SchedulingBlockContainer(
                sb_id=np.uint64(self.sb_id),
                sb_type=SchedulingBlockType.OBSERVATION,
                producer_id="simulation",
                observing_mode=ObservingMode.UNKNOWN,
                pointing_mode=PointingMode.DRIFT,
            )
        }

        ob_dict = {
            self.obs_id: ObservationBlockContainer(
                obs_id=np.uint64(self.obs_id),
                sb_id=np.uint64(self.sb_id),
                producer_id="simulation",
                state=ObservationBlockState.COMPLETED_SUCCEDED,
                subarray_pointing_lat=alt * u.rad,
                subarray_pointing_lon=az * u.rad,
                subarray_pointing_frame=CoordinateFrameType.ALTAZ,
                actual_start_time=Time(self.file_.header["time"], format="unix"),
                scheduled_start_time=Time(self.file_.header["time"], format="unix"),
            )
        }

        return sb_dict, ob_dict

    @staticmethod
    def _fill_simulated_event_information(array_event):
        mc_event = array_event["mc_event"]
        mc_shower = array_event["mc_shower"]
        if mc_shower is None:
            return

        alt = _clip_altitude_if_close(mc_shower["altitude"])

        return SimulatedShowerContainer(
            energy=u.Quantity(mc_shower["energy"], u.TeV),
            alt=Angle(alt, u.rad),
            az=Angle(mc_shower["azimuth"], u.rad),
            core_x=u.Quantity(mc_event["xcore"], u.m),
            core_y=u.Quantity(mc_event["ycore"], u.m),
            h_first_int=u.Quantity(mc_shower["h_first_int"], u.m),
            x_max=u.Quantity(mc_shower["xmax"], GRAMMAGE_UNIT),
            shower_primary_id=mc_shower["primary_id"],
            starting_grammage=u.Quantity(
                mc_shower.get("depth_start", _float32_nan), GRAMMAGE_UNIT
            ),
        )
