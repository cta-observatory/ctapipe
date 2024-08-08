"""
Container structures for data that should be read or written to disk
"""

import enum
from functools import partial

import numpy as np
from astropy import units as u
from astropy.time import Time
from numpy import nan

from .core import Container, Field, Map

__all__ = [
    "ArrayEventContainer",
    "ConcentrationContainer",
    "DL0CameraContainer",
    "DL0Container",
    "DL1CameraCalibrationContainer",
    "DL1CameraContainer",
    "DL1Container",
    "DL2Container",
    "EventCalibrationContainer",
    "EventCameraCalibrationContainer",
    "EventIndexContainer",
    "EventType",
    "FlatFieldContainer",
    "HillasParametersContainer",
    "CoreParametersContainer",
    "ImageParametersContainer",
    "LeakageContainer",
    "MonitoringCameraContainer",
    "MonitoringContainer",
    "MorphologyContainer",
    "BaseHillasParametersContainer",
    "CameraHillasParametersContainer",
    "CameraTimingParametersContainer",
    "ParticleClassificationContainer",
    "PedestalContainer",
    "PixelStatusContainer",
    "R0CameraContainer",
    "R0Container",
    "R1CameraContainer",
    "R1Container",
    "ReconstructedContainer",
    "ReconstructedEnergyContainer",
    "ReconstructedGeometryContainer",
    "DispContainer",
    "SimulatedCameraContainer",
    "SimulatedShowerContainer",
    "SimulatedShowerDistribution",
    "SimulationConfigContainer",
    "TelEventIndexContainer",
    "BaseTimingParametersContainer",
    "TimingParametersContainer",
    "TriggerContainer",
    "WaveformCalibrationContainer",
    "StatisticsContainer",
    "ImageStatisticsContainer",
    "IntensityStatisticsContainer",
    "PeakTimeStatisticsContainer",
    "SchedulingBlockContainer",
    "ObservationBlockContainer",
    "ObservingMode",
    "ObservationBlockState",
]


# see https://github.com/astropy/astropy/issues/6509
NAN_TIME = Time(0, format="mjd", scale="tai")

#: Used for unsigned integer obs_id or sb_id default values:
UNKNOWN_ID = np.uint64(np.iinfo(np.uint64).max)
#: Used for unsigned integer tel_id default value
UNKNOWN_TEL_ID = np.uint16(np.iinfo(np.uint16).max)


obs_id_field = partial(Field, UNKNOWN_ID, description="Observation Block ID")
event_id_field = partial(Field, UNKNOWN_ID, description="Array Event ID")
tel_id_field = partial(Field, UNKNOWN_TEL_ID, description="Telescope ID")


class SchedulingBlockType(enum.Enum):
    """
    Types of Scheduling Block
    """

    UNKNOWN = -1
    OBSERVATION = 0
    CALIBRATION = 1
    ENGINEERING = 2


class ObservationBlockState(enum.Enum):
    """Observation Block States. Part of the Observation Configuration data
    model.
    """

    UNKNOWN = -1
    FAILED = 0
    COMPLETED_SUCCEDED = 1
    COMPLETED_CANCELED = 2
    COMPLETED_TRUNCATED = 3
    ARCHIVED = 4


class ObservingMode(enum.Enum):
    """How a scheduling block is observed. Part of the Observation Configuration
    data model.

    """

    UNKNOWN = -1
    WOBBLE = 0
    ON_OFF = 1
    GRID = 2
    CUSTOM = 3


class PointingMode(enum.Enum):
    """Describes how the telescopes move. Part of the Observation Configuration
    data model.

    """

    UNKNOWN = -1
    #: drives track a point that moves with the sky
    TRACK = 0
    #: drives stay fixed at an alt/az point while the sky drifts by
    DRIFT = 1


class CoordinateFrameType(enum.Enum):
    """types of coordinate frames used in ObservationBlockContainers. Part of
    the Observation Configuration data model.

    """

    UNKNOWN = -1
    ALTAZ = 0
    ICRS = 1
    GALACTIC = 2


class EventType(enum.Enum):
    """Enum of EventTypes as defined in :cite:p:`ctao-r1-event-data-model`"""

    # calibrations are 0-15
    FLATFIELD = 0
    SINGLE_PE = 1
    SKY_PEDESTAL = 2
    DARK_PEDESTAL = 3
    ELECTRONIC_PEDESTAL = 4
    OTHER_CALIBRATION = 15

    #: For mono-telescope triggers (not used in MC)
    MUON = 16
    HARDWARE_STEREO = 17

    #: ACADA (DAQ) software trigger
    DAQ = 24

    #: Standard Physics  stereo trigger
    SUBARRAY = 32

    UNKNOWN = 255


class VarianceType(enum.Enum):
    """Enum of variance types used for the VarianceContainer"""

    # Simple variance of waveform
    WAVEFORM = 0
    # Variance of integrated samples of a waveform
    INTEGRATED = 1


class PixelStatus(enum.IntFlag):
    """
    Pixel status information

    See DL0 Data Model specification:
    https://redmine.cta-observatory.org/dmsf/files/17552/view
    """

    DVR_STORED_AS_SIGNAL = enum.auto()
    DVR_STORED_NO_SIGNAL = enum.auto()
    HIGH_GAIN_STORED = enum.auto()
    LOW_GAIN_STORED = enum.auto()
    SATURATED = enum.auto()
    PIXEL_TRIGGER_0 = enum.auto()
    PIXEL_TRIGGER_1 = enum.auto()
    PIXEL_TRIGGER_2 = enum.auto()

    #: DVR status uses two bits
    #: 0 = not stored, 1 = identified as signal, 2 = stored, not identified as signal
    DVR_STATUS = DVR_STORED_AS_SIGNAL | DVR_STORED_NO_SIGNAL

    #: Pixel trigger information, TBD
    PIXEL_TRIGGER = PIXEL_TRIGGER_0 | PIXEL_TRIGGER_1 | PIXEL_TRIGGER_2

    @staticmethod
    def get_dvr_status(pixel_status):
        """
        Return only the bits corresponding to the DVR_STATUS

        Returns
        -------
        dvr_status: int or array[uint8]
            0 = pixel not stored
            1 = pixel was identified as signal pixel and stored
            2 = pixel was stored, but not identified as signal
        """
        return pixel_status & PixelStatus.DVR_STATUS

    @staticmethod
    def get_channel_info(pixel_status):
        """
        Return only the bits corresponding to the channel info (high/low gain stored)

        Returns
        -------
        channel_info: int or array[uint8]
            0 = pixel broken / disabled
            1 = only high gain read out
            2 = only low gain read out
            3 = both gains read out
        """
        gain_bits = PixelStatus.HIGH_GAIN_STORED | PixelStatus.LOW_GAIN_STORED
        return (pixel_status & gain_bits) >> 2

    @staticmethod
    def is_invalid(pixel_status):
        """Return if pixel values are marked as invalid

        This is encoded in the data model as neither high gain nor low gain marked as stored
        """
        gain_bits = PixelStatus.HIGH_GAIN_STORED | PixelStatus.LOW_GAIN_STORED
        return (pixel_status & gain_bits) == 0


class TelescopeConfigurationIndexContainer(Container):
    """Index to include for per-OB telescope configuration"""

    default_prefix = ""
    obs_id = obs_id_field()
    tel_id = tel_id_field()


class EventIndexContainer(Container):
    """Index columns to include in event lists.

    Common to all data levels
    """

    default_prefix = ""
    obs_id = obs_id_field()
    event_id = event_id_field()


class TelEventIndexContainer(Container):
    """
    index columns to include in telescope-wise event lists

    Common to all data levels that have telescope-wise information.
    """

    default_prefix = ""
    obs_id = obs_id_field()
    event_id = event_id_field()
    tel_id = tel_id_field()


class BaseHillasParametersContainer(Container):
    """
    Base container for hillas parameters to
    allow the CameraHillasParametersContainer to
    be assigned to an ImageParametersContainer as well.
    """

    intensity = Field(nan, "total intensity (size)")
    skewness = Field(nan, "measure of the asymmetry")
    kurtosis = Field(nan, "measure of the tailedness")


class CameraHillasParametersContainer(BaseHillasParametersContainer):
    """
    Hillas Parameters in the camera frame. The cog position
    is given in meter from the camera center.
    """

    default_prefix = "camera_frame_hillas"
    x = Field(nan * u.m, "centroid x coordinate", unit=u.m)
    y = Field(nan * u.m, "centroid x coordinate", unit=u.m)
    r = Field(nan * u.m, "radial coordinate of centroid", unit=u.m)
    phi = Field(nan * u.deg, "polar coordinate of centroid", unit=u.deg)

    length = Field(nan * u.m, "standard deviation along the major-axis", unit=u.m)
    length_uncertainty = Field(nan * u.m, "uncertainty of length", unit=u.m)
    width = Field(nan * u.m, "standard spread along the minor-axis", unit=u.m)
    width_uncertainty = Field(nan * u.m, "uncertainty of width", unit=u.m)
    psi = Field(nan * u.deg, "rotation angle of ellipse", unit=u.deg)


class HillasParametersContainer(BaseHillasParametersContainer):
    """
    Hillas Parameters in a spherical system centered on the pointing position
    (TelescopeFrame). The cog position is given as offset in
    longitude and latitude in degree.
    """

    default_prefix = "hillas"
    fov_lon = Field(
        nan * u.deg,
        "longitude angle in a spherical system centered on the pointing position",
        unit=u.deg,
    )
    fov_lat = Field(
        nan * u.deg,
        "latitude angle in a spherical system centered on the pointing position",
        unit=u.deg,
    )
    r = Field(nan * u.deg, "radial coordinate of centroid", unit=u.deg)
    phi = Field(nan * u.deg, "polar coordinate of centroid", unit=u.deg)

    length = Field(nan * u.deg, "standard deviation along the major-axis", unit=u.deg)
    length_uncertainty = Field(nan * u.deg, "uncertainty of length", unit=u.deg)
    width = Field(nan * u.deg, "standard spread along the minor-axis", unit=u.deg)
    width_uncertainty = Field(nan * u.deg, "uncertainty of width", unit=u.deg)
    psi = Field(nan * u.deg, "rotation angle of ellipse", unit=u.deg)


class LeakageContainer(Container):
    """
    Fraction of signal in 1 or 2-pixel width border from the edge of the
    camera, measured in number of signal pixels or in intensity.
    """

    default_prefix = "leakage"

    pixels_width_1 = Field(
        nan, "fraction of pixels after cleaning that are in camera border of width=1"
    )
    pixels_width_2 = Field(
        nan, "fraction of pixels after cleaning that are in camera border of width=2"
    )
    intensity_width_1 = Field(
        np.float32(nan),
        "Intensity in photo-electrons after cleaning"
        " that are in the camera border of width=1 pixel",
    )
    intensity_width_2 = Field(
        np.float32(nan),
        "Intensity in photo-electrons after cleaning"
        " that are in the camera border of width=2 pixels",
    )


class ConcentrationContainer(Container):
    """
    Concentrations are ratios between light amount
    in certain areas of the image and the full image.
    """

    default_prefix = "concentration"
    cog = Field(
        nan, "Percentage of photo-electrons inside one pixel diameter of the cog"
    )
    core = Field(nan, "Percentage of photo-electrons inside the hillas ellipse")
    pixel = Field(nan, "Percentage of photo-electrons in the brightest pixel")


class BaseTimingParametersContainer(Container):
    """
    Base container for timing parameters to
    allow the CameraTimingParametersContainer to
    be assigned to an ImageParametersContainer as well.
    """

    intercept = Field(nan, "intercept of arrival times along main shower axis")
    deviation = Field(
        nan,
        "Root-mean-square deviation of the pulse times "
        "with respect to the predicted time",
    )


class CameraTimingParametersContainer(BaseTimingParametersContainer):
    """
    Slope and Intercept of a linear regression of the arrival times
    along the shower main axis in the camera frame.
    """

    default_prefix = "camera_frame_timing"
    slope = Field(
        nan / u.m, "Slope of arrival times along main shower axis", unit=1 / u.m
    )


class TimingParametersContainer(BaseTimingParametersContainer):
    """
    Slope and Intercept of a linear regression of the arrival times
    along the shower main axis in a
    spherical system centered on the pointing position (TelescopeFrame)
    """

    default_prefix = "timing"
    slope = Field(
        nan / u.deg, "Slope of arrival times along main shower axis", unit=1 / u.deg
    )


class MorphologyContainer(Container):
    """Parameters related to pixels surviving image cleaning"""

    n_pixels = Field(-1, "Number of usable pixels")
    n_islands = Field(-1, "Number of distinct islands in the image")
    n_small_islands = Field(-1, "Number of <= 2 pixel islands")
    n_medium_islands = Field(-1, "Number of 2-50 pixel islands")
    n_large_islands = Field(-1, "Number of > 50 pixel islands")


class StatisticsContainer(Container):
    """Store descriptive statistics of a chunk of images"""

    n_events = Field(-1, "number of events used for the extraction of the statistics")
    mean = Field(
        None,
        "mean of a pixel-wise quantity for each channel"
        "Type: float; Shape: (n_channels, n_pixel)",
    )
    median = Field(
        None,
        "median of a pixel-wise quantity for each channel"
        "Type: float; Shape: (n_channels, n_pixel)",
    )
    std = Field(
        None,
        "standard deviation of a pixel-wise quantity for each channel"
        "Type: float; Shape: (n_channels, n_pixel)",
    )


class ImageStatisticsContainer(Container):
    """Store descriptive image statistics"""

    max = Field(np.float32(nan), "value of pixel with maximum intensity")
    min = Field(np.float32(nan), "value of pixel with minimum intensity")
    mean = Field(np.float32(nan), "mean intensity")
    std = Field(np.float32(nan), "standard deviation of intensity")
    skewness = Field(nan, "skewness of intensity")
    kurtosis = Field(nan, "kurtosis of intensity")


class IntensityStatisticsContainer(ImageStatisticsContainer):
    default_prefix = "intensity"


class PeakTimeStatisticsContainer(ImageStatisticsContainer):
    default_prefix = "peak_time"


class CoreParametersContainer(Container):
    """Telescope-wise shower's direction in the Tilted/Ground Frame"""

    default_prefix = "core"
    psi = Field(nan * u.deg, "Image direction in the Tilted/Ground Frame", unit="deg")


class ImageParametersContainer(Container):
    """Collection of image parameters"""

    default_prefix = "params"
    hillas = Field(
        default_factory=HillasParametersContainer,
        description="Hillas Parameters",
        type=BaseHillasParametersContainer,
    )
    timing = Field(
        default_factory=TimingParametersContainer,
        description="Timing Parameters",
        type=BaseTimingParametersContainer,
    )
    leakage = Field(
        default_factory=LeakageContainer,
        description="Leakage Parameters",
    )
    concentration = Field(
        default_factory=ConcentrationContainer,
        description="Concentration Parameters",
    )
    morphology = Field(
        default_factory=MorphologyContainer, description="Image Morphology Parameters"
    )
    intensity_statistics = Field(
        default_factory=IntensityStatisticsContainer,
        description="Intensity image statistics",
    )
    peak_time_statistics = Field(
        default_factory=PeakTimeStatisticsContainer,
        description="Peak time image statistics",
    )
    core = Field(
        default_factory=CoreParametersContainer,
        description="Image direction in the Tilted/Ground Frame",
    )


class DL1CameraContainer(Container):
    """
    Storage of output of camera calibration e.g the final calibrated
    image in intensity units and the pulse time.
    """

    image = Field(
        None,
        "Numpy array of camera image, after waveform extraction."
        "Shape: (n_pixel) if n_channels is 1 or data is gain selected"
        "else: (n_channels, n_pixel)",
    )
    peak_time = Field(
        None,
        "Numpy array containing position of the peak of the pulse as determined by "
        "the extractor."
        "Shape: (n_pixel) if n_channels is 1 or data is gain selected"
        "else: (n_channels, n_pixel)",
    )
    image_mask = Field(
        None,
        "Boolean numpy array where True means the pixel has passed cleaning."
        " Shape: (n_pixel, )",
        dtype=np.bool_,
        ndim=1,
    )
    is_valid = Field(
        False,
        (
            "True if image extraction succeeded, False if failed "
            "or in the case of TwoPass methods, that the first "
            "pass only was returned."
        ),
    )
    parameters = Field(
        None, description="Image parameters", type=ImageParametersContainer
    )


class DL1Container(Container):
    """DL1 Calibrated Camera Images and associated data"""

    tel = Field(
        default_factory=partial(Map, DL1CameraContainer),
        description="map of tel_id to DL1CameraContainer",
    )


class DL1CameraCalibrationContainer(Container):
    """
    Storage of DL1 calibration parameters for the current event
    """

    pedestal_offset = Field(
        None,
        "Residual mean pedestal of the waveforms for each pixel."
        " This value is subtracted from the waveforms of each pixel before"
        " the pulse extraction. Shape: (n_channels, n_pixels)",
    )
    absolute_factor = Field(
        None,
        "Multiplicative coefficients for the absolute calibration of extracted charge"
        " into physical units (e.g. photoelectrons or photons) for each pixel."
        " Shape: (n_channels, n_pixels)",
    )
    relative_factor = Field(
        None,
        "Multiplicative Coefficients for the relative correction between pixels to"
        " achieve a uniform charge response (post absolute calibration) from a"
        " uniform illumination. Shape: (n_channels, n_pixels)",
    )
    time_shift = Field(
        None,
        "Additive coefficients for the timing correction before charge extraction"
        " for each pixel. Shape: (n_channels, n_pixels)",
    )


class R0CameraContainer(Container):
    """
    Storage of raw data from a single telescope
    """

    waveform = Field(
        None, ("numpy array containing ADC samples: (n_channels, n_pixels, n_samples)")
    )


class R0Container(Container):
    """
    Storage of a Merged Raw Data Event
    """

    tel = Field(
        default_factory=partial(Map, R0CameraContainer),
        description="map of tel_id to R0CameraContainer",
    )


class R1CameraContainer(Container):
    """
    Storage of r1 calibrated data from a single telescope
    """

    event_type = Field(EventType.UNKNOWN, "type of event", type=EventType)
    event_time = Field(NAN_TIME, "event timestamp")

    waveform = Field(
        None,
        (
            "numpy array containing a set of images, one per ADC sample"
            "Shape: (n_channels, n_pixels, n_samples)"
        ),
    )

    pixel_status = Field(
        None,
        "Array of pixel status values, see PixelStatus for definition of the values",
        ndim=1,
        dtype=np.uint8,
    )

    first_cell_id = Field(
        None,
        "Array of first cell ids of the readout chips. Only used by LST and SST.",
        dtype=np.uint16,
        ndim=1,
    )

    module_hires_local_clock_counter = Field(
        None,
        "Clock counter values of the camera modules. See R1 data model for details.",
        dtype=np.uint64,
        ndim=1,
    )

    pedestal_intensity = Field(
        None,
        "Pedestal intensity in each pixel in DC",
        dtype=np.float32,
        ndim=1,
    )

    calibration_monitoring_id = Field(
        None,
        "ID of the CalibrationMonitoringSet containing the applied pre-calibration parameters",
    )

    selected_gain_channel = Field(
        None,
        (
            "Numpy array containing the gain channel chosen for each pixel. "
            "Note: should be replaced by using ``pixel_status`` "
            "Shape: (n_pixels)"
        ),
    )


class R1Container(Container):
    """
    Storage of a r1 calibrated Data Event
    """

    tel = Field(
        default_factory=partial(Map, R1CameraContainer),
        description="map of tel_id to R1CameraContainer",
    )


class DL0CameraContainer(Container):
    """
    Storage of data volume reduced dl0 data from a single telescope.

    See DL0 Data Model specification:
    https://redmine.cta-observatory.org/dmsf/files/17552/view
    """

    event_type = Field(EventType.UNKNOWN, "type of event", type=EventType)
    event_time = Field(NAN_TIME, "event timestamp")

    waveform = Field(
        None,
        (
            "numpy array containing data volume reduced "
            "p.e. samples"
            "(n_channels, n_pixels, n_samples). Note this may be a masked array, "
            "if pixels or time slices are zero-suppressed"
        ),
    )

    pixel_status = Field(
        None,
        "Array of pixel status values, see PixelStatus for definition of the values",
        dtype=np.uint8,
        ndim=1,
    )

    first_cell_id = Field(
        None,
        "Array of first cell ids of the readout chips. Only used by LST and SST.",
        dtype=np.uint16,
        ndim=1,
    )

    calibration_monitoring_id = Field(
        None,
        "ID of the CalibrationMonitoringSet containing the applied pre-calibration parameters",
    )

    selected_gain_channel = Field(
        None,
        (
            "Numpy array containing the gain channel chosen for each pixel. "
            "Note: this should be replaced by only using ``pixel_status`` "
            "Shape: (n_pixels)"
        ),
    )


class DL0Container(Container):
    """
    Storage of a data volume reduced Event
    """

    tel = Field(
        default_factory=partial(Map, DL0CameraContainer),
        description="map of tel_id to DL0CameraContainer",
    )


class TelescopeImpactParameterContainer(Container):
    """
    Impact Parameter computed from reconstructed shower geometry
    """

    default_prefix = "impact"

    distance = Field(
        nan * u.m, "distance of the telescope to the shower axis", unit=u.m
    )
    distance_uncert = Field(nan * u.m, "uncertainty in impact_parameter", unit=u.m)


class SimulatedShowerContainer(Container):
    default_prefix = "true"
    energy = Field(nan * u.TeV, "Simulated Energy", unit=u.TeV)
    alt = Field(nan * u.deg, "Simulated altitude", unit=u.deg)
    az = Field(nan * u.deg, "Simulated azimuth", unit=u.deg)
    core_x = Field(nan * u.m, "Simulated core position (x)", unit=u.m)
    core_y = Field(nan * u.m, "Simulated core position (y)", unit=u.m)
    h_first_int = Field(nan * u.m, "Height of first interaction", unit=u.m)
    x_max = Field(nan * u.g / u.cm**2, "Simulated Xmax value", unit=u.g / u.cm**2)
    starting_grammage = Field(
        nan * u.g / u.cm**2,
        "Grammage (mass overburden) where the particle was injected into the atmosphere",
        unit=u.g / u.cm**2,
    )
    shower_primary_id = Field(
        np.int16(np.iinfo(np.int16).max),
        "Simulated shower primary ID 0 (gamma), 1(e-),"
        "2(mu-), 100*A+Z for nucleons and nuclei,"
        "negative for antimatter.",
    )


class SimulatedCameraContainer(Container):
    """
    True images and parameters derived from them, analogous to the `DL1CameraContainer`
    but for simulated data.
    """

    default_prefix = ""

    true_image_sum = Field(
        np.int32(-1), "Total number of detected Cherenkov photons in the camera"
    )
    true_image = Field(
        None,
        "Numpy array of camera image in PE as simulated before noise has been added. "
        "Shape: (n_pixel)",
        dtype=np.int32,
        ndim=1,
    )

    true_parameters = Field(
        None,
        description="Parameters derived from the true_image",
        type=ImageParametersContainer,
    )

    impact = Field(
        default_factory=TelescopeImpactParameterContainer,
        description="true impact parameter",
    )


class SimulatedEventContainer(Container):
    shower = Field(
        default_factory=SimulatedShowerContainer,
        description="True event information",
    )
    tel = Field(default_factory=partial(Map, SimulatedCameraContainer))


class SimulationConfigContainer(Container):
    """
    Configuration parameters of the simulation
    """

    run_number = Field(np.int32(-1), description="Original sim_telarray run number")
    corsika_version = Field(nan, description="CORSIKA version * 1000")
    simtel_version = Field(nan, description="sim_telarray version * 1000")
    energy_range_min = Field(
        nan * u.TeV,
        description="Lower limit of energy range of primary particle",
        unit=u.TeV,
    )
    energy_range_max = Field(
        nan * u.TeV,
        description="Upper limit of energy range of primary particle",
        unit=u.TeV,
    )
    prod_site_B_total = Field(
        nan * u.uT, description="total geomagnetic field", unit=u.uT
    )
    prod_site_B_declination = Field(
        nan * u.rad, description="magnetic declination", unit=u.rad
    )
    prod_site_B_inclination = Field(
        nan * u.rad, description="magnetic inclination", unit=u.rad
    )
    prod_site_alt = Field(
        nan * u.m, description="height of observation level", unit=u.m
    )
    spectral_index = Field(nan, description="Power-law spectral index of spectrum")
    shower_prog_start = Field(
        nan, description="Time when shower simulation started, CORSIKA: only date"
    )
    shower_prog_id = Field(nan, description="CORSIKA=1, ALTAI=2, KASCADE=3, MOCCA=4")
    detector_prog_start = Field(
        nan, description="Time when detector simulation started"
    )
    detector_prog_id = Field(nan, description="simtelarray=1")
    n_showers = Field(nan, description="Number of showers simulated")
    shower_reuse = Field(nan, description="Numbers of uses of each shower")
    max_alt = Field(nan * u.rad, description="Maximum shower altitude", unit=u.rad)
    min_alt = Field(nan * u.rad, description="Minimum shower altitude", unit=u.rad)
    max_az = Field(nan * u.rad, description="Maximum shower azimuth", unit=u.rad)
    min_az = Field(nan * u.rad, description="Minimum shower azimuth", unit=u.rad)
    diffuse = Field(False, description="Diffuse Mode On/Off")
    max_viewcone_radius = Field(
        nan * u.deg, description="Maximum viewcone radius", unit=u.deg
    )
    min_viewcone_radius = Field(
        nan * u.deg, description="Minimum viewcone radius", unit=u.deg
    )
    max_scatter_range = Field(nan * u.m, description="Maximum scatter range", unit=u.m)
    min_scatter_range = Field(nan * u.m, description="Minimum scatter range", unit=u.m)
    core_pos_mode = Field(
        nan, description="Core Position Mode (0=Circular, 1=Rectangular)"
    )
    atmosphere = Field(nan * u.m, description="Atmospheric model number")
    corsika_iact_options = Field(
        nan, description="CORSIKA simulation options for IACTs"
    )
    corsika_low_E_model = Field(
        nan, description="CORSIKA low-energy simulation physics model"
    )
    corsika_high_E_model = Field(
        nan,
        "CORSIKA physics model ID for high energies "
        "(1=VENUS, 2=SIBYLL, 3=QGSJET, 4=DPMJET, 5=NeXus, 6=EPOS) ",
    )
    corsika_bunchsize = Field(nan, description="Number of Cherenkov photons per bunch")
    corsika_wlen_min = Field(
        nan * u.m, description="Minimum wavelength of cherenkov light", unit=u.nm
    )
    corsika_wlen_max = Field(
        nan * u.m, description="Maximum wavelength of cherenkov light", unit=u.nm
    )
    corsika_low_E_detail = Field(
        nan, description="More details on low E interaction model (version etc.)"
    )
    corsika_high_E_detail = Field(
        nan, description="More details on high E interaction model (version etc.)"
    )


class TelescopeTriggerContainer(Container):
    default_prefix = ""
    time = Field(NAN_TIME, description="Telescope trigger time")
    n_trigger_pixels = Field(
        -1, description="Number of trigger groups (sectors) listed"
    )
    trigger_pixels = Field(None, description="pixels involved in the camera trigger")


class TriggerContainer(Container):
    default_prefix = ""
    time = Field(NAN_TIME, description="central average time stamp")
    tels_with_trigger = Field(
        None, description="List of telescope ids that triggered the array event"
    )
    event_type = Field(EventType.SUBARRAY, description="Event type")
    tel = Field(
        default_factory=partial(Map, TelescopeTriggerContainer),
        description="telescope-wise trigger information",
    )


class ReconstructedGeometryContainer(Container):
    """
    Standard output of algorithms reconstructing shower geometry
    """

    default_prefix = ""

    alt = Field(nan * u.deg, "reconstructed altitude", unit=u.deg)
    alt_uncert = Field(nan * u.deg, "reconstructed altitude uncertainty", unit=u.deg)
    az = Field(nan * u.deg, "reconstructed azimuth", unit=u.deg)
    az_uncert = Field(nan * u.deg, "reconstructed azimuth uncertainty", unit=u.deg)
    ang_distance_uncert = Field(
        nan * u.deg,
        "uncertainty radius of reconstructed altitude-azimuth position",
        unit=u.deg,
    )
    core_x = Field(
        nan * u.m, "reconstructed x coordinate of the core position", unit=u.m
    )
    core_y = Field(
        nan * u.m, "reconstructed y coordinate of the core position", unit=u.m
    )
    core_uncert_x = Field(
        nan * u.m,
        "reconstructed core position uncertainty along ground frame X axis",
        unit=u.m,
    )
    core_uncert_y = Field(
        nan * u.m,
        "reconstructed core position uncertainty along ground frame Y axis",
        unit=u.m,
    )
    core_tilted_x = Field(
        nan * u.m, "reconstructed x coordinate of the core position", unit=u.m
    )
    core_tilted_y = Field(
        nan * u.m, "reconstructed y coordinate of the core position", unit=u.m
    )
    core_tilted_uncert_x = Field(
        nan * u.m,
        "reconstructed core position uncertainty along tilted frame X axis",
        unit=u.m,
    )
    core_tilted_uncert_y = Field(
        nan * u.m,
        "reconstructed core position uncertainty along tilted frame Y axis",
        unit=u.m,
    )
    h_max = Field(
        nan * u.m,
        "reconstructed vertical height above sea level of the shower maximum",
        unit=u.m,
    )
    h_max_uncert = Field(nan * u.m, "uncertainty of h_max", unit=u.m)

    is_valid = Field(
        False,
        (
            "Geometry validity flag. True if the shower geometry"
            "was properly reconstructed by the algorithm"
        ),
    )
    average_intensity = Field(
        nan, "average intensity of the intensities used for reconstruction"
    )
    goodness_of_fit = Field(nan, "measure of algorithm success (if fit)")
    telescopes = Field(None, "Telescopes used if stereo, or None if Mono")


class ReconstructedEnergyContainer(Container):
    """
    Standard output of algorithms estimating energy
    """

    default_prefix = ""

    energy = Field(nan * u.TeV, "reconstructed energy", unit=u.TeV)
    energy_uncert = Field(nan * u.TeV, "reconstructed energy uncertainty", unit=u.TeV)
    is_valid = Field(
        False,
        (
            "energy reconstruction validity flag. True if "
            "the energy was properly reconstructed by the "
            "algorithm"
        ),
    )
    goodness_of_fit = Field(nan, "goodness of the algorithm fit")
    telescopes = Field(None, "Telescopes used if stereo, or None if Mono")


class ParticleClassificationContainer(Container):
    """
    Standard output of gamma/hadron classification algorithms
    """

    default_prefix = ""

    prediction = Field(
        nan,
        (
            " prediction of the classifier, defined between [0,1]"
            ", where values close to 1 mean that the positive class"
            " (e.g. gamma in gamma-ray analysis) is more likely"
        ),
    )
    is_valid = Field(False, "true if classification parameters are valid")
    goodness_of_fit = Field(nan, "goodness of the algorithm fit")
    telescopes = Field(None, "Telescopes used if stereo, or None if Mono")


class DispContainer(Container):
    """
    Standard output of disp reconstruction algorithms for origin reconstruction
    """

    default_prefix = "disp"

    parameter = Field(
        nan * u.deg, "reconstructed value for disp (= sign * norm)", unit=u.deg
    )
    sign_score = Field(
        nan,
        "Score for how certain the disp sign classification was."
        " 0 means completely uncertain, 1 means very certain.",
    )


class ReconstructedContainer(Container):
    """Reconstructed shower info from multiple algorithms"""

    # Note: there is a reason why the hiererchy is
    # `event.dl2.stereo.geometry[algorithm]` and not
    # `event.dl2[algorithm].stereo.geometry` and that is because when writing
    # the data, the former makes it easier to only write information that a
    # particular reconstructor generates, e.g. only write the geometry in cases
    # where energy is not yet computed. Some algorithms will compute all three,
    # but most will compute only fill or two of these sub-Contaiers:

    geometry = Field(
        default_factory=partial(Map, ReconstructedGeometryContainer),
        description="map of algorithm to reconstructed shower parameters",
    )
    energy = Field(
        default_factory=partial(Map, ReconstructedEnergyContainer),
        description="map of algorithm to reconstructed energy parameters",
    )
    classification = Field(
        default_factory=partial(Map, ParticleClassificationContainer),
        description="map of algorithm to classification parameters",
    )


class TelescopeReconstructedContainer(ReconstructedContainer):
    """Telescope-wise reconstructed quantities"""

    impact = Field(
        default_factory=partial(Map, TelescopeImpactParameterContainer),
        description="map of algorithm to impact parameter info",
    )
    disp = Field(
        default_factory=partial(Map, DispContainer),
        description="map of algorithm to reconstructed disp parameters",
    )


class DL2Container(Container):
    """Reconstructed Shower information for a given reconstruction algorithm,
    including optionally both per-telescope mono reconstruction and per-shower
    stereo reconstructions
    """

    tel = Field(
        default_factory=partial(Map, TelescopeReconstructedContainer),
        description="map of tel_id to single-telescope reconstruction (DL2a)",
    )
    stereo = Field(
        default_factory=ReconstructedContainer,
        description="Stereo Shower reconstruction results",
    )


class TelescopePointingContainer(Container):
    """
    Container holding pointing information for a single telescope
    after all necessary correction and calibration steps.
    These values should be used in the reconstruction to transform
    between camera and sky coordinates.
    """

    default_prefix = "telescope_pointing"
    azimuth = Field(nan * u.rad, "Azimuth, measured N->E", unit=u.rad)
    altitude = Field(nan * u.rad, "Altitude", unit=u.rad)


class PointingContainer(Container):
    tel = Field(
        default_factory=partial(Map, TelescopePointingContainer),
        description="Telescope pointing positions",
    )
    array_azimuth = Field(nan * u.rad, "Array pointing azimuth", unit=u.rad)
    array_altitude = Field(nan * u.rad, "Array pointing altitude", unit=u.rad)
    array_ra = Field(nan * u.rad, "Array pointing right ascension", unit=u.rad)
    array_dec = Field(nan * u.rad, "Array pointing declination", unit=u.rad)


class EventCameraCalibrationContainer(Container):
    """
    Container for the calibration coefficients for the current event and camera
    """

    dl1 = Field(
        default_factory=DL1CameraCalibrationContainer,
        description="Container for DL1 calibration coefficients",
    )


class EventCalibrationContainer(Container):
    """
    Container for calibration coefficients for the current event
    """

    # create the camera container
    tel = Field(
        default_factory=partial(Map, EventCameraCalibrationContainer),
        description="map of tel_id to EventCameraCalibrationContainer",
    )


class MuonRingContainer(Container):
    """Container for the result of a ring fit in telescope frame"""

    center_fov_lon = Field(
        nan * u.deg, "center (fov_lon) of the fitted muon ring", unit=u.deg
    )
    center_fov_lat = Field(
        nan * u.deg, "center (fov_lat) of the fitted muon ring", unit=u.deg
    )
    radius = Field(nan * u.deg, "radius of the fitted muon ring", unit=u.deg)
    center_phi = Field(
        nan * u.deg, "Angle of ring center within camera plane", unit=u.deg
    )
    center_distance = Field(
        nan * u.deg, "Distance of ring center from camera center", unit=u.deg
    )


class MuonEfficiencyContainer(Container):
    width = Field(nan * u.deg, "width of the muon ring in degrees")
    impact = Field(nan * u.m, "distance of muon impact position from center of mirror")
    impact_x = Field(nan * u.m, "impact parameter x position")
    impact_y = Field(nan * u.m, "impact parameter y position")
    optical_efficiency = Field(nan, "optical efficiency muon")
    is_valid = Field(False, "True if the fit converged successfully")
    parameters_at_limit = Field(
        False, "True if any bounded parameter was fitted close to a bound"
    )
    likelihood_value = Field(nan, "cost function value at the minimum")


class MuonParametersContainer(Container):
    containment = Field(nan, "containment of the ring inside the camera")
    completeness = Field(
        nan,
        "Complenetess of the muon ring"
        ", estimated by dividing the ring into segments"
        " and counting segments above a threshold",
    )
    intensity_ratio = Field(nan, "Intensity ratio of pixels in the ring to all pixels")
    mean_squared_error = Field(
        nan * u.deg**2,
        "MSE of the deviation of all pixels after cleaning from the ring fit",
    )


class MuonTelescopeContainer(Container):
    """
    Container for muon analysis
    """

    ring = Field(default_factory=MuonRingContainer, description="muon ring fit")
    parameters = Field(
        default_factory=MuonParametersContainer, description="muon parameters"
    )
    efficiency = Field(
        default_factory=MuonEfficiencyContainer, description="muon efficiency"
    )


class MuonContainer(Container):
    """Root container for muon parameters"""

    tel = Field(
        default_factory=partial(Map, MuonTelescopeContainer),
        description="map of tel_id to MuonTelescopeContainer",
    )


class FlatFieldContainer(Container):
    """
    Container for flat-field parameters obtained from a set of
    [n_events] flat-field events
    """

    sample_time = Field(
        0 * u.s, "Time associated to the flat-field event set ", unit=u.s
    )
    sample_time_min = Field(
        nan * u.s, "Minimum time of the flat-field events", unit=u.s
    )
    sample_time_max = Field(
        nan * u.s, "Maximum time of the flat-field events", unit=u.s
    )
    n_events = Field(0, "Number of events used for statistics")

    charge_mean = Field(None, "np array of signal charge mean (n_chan, n_pix)")
    charge_median = Field(None, "np array of signal charge median (n_chan, n_pix)")
    charge_std = Field(
        None, "np array of signal charge standard deviation (n_chan, n_pix)"
    )
    time_mean = Field(None, "np array of signal time mean (n_chan, n_pix)", unit=u.ns)
    time_median = Field(
        None, "np array of signal time median (n_chan, n_pix)", unit=u.ns
    )
    time_std = Field(
        None, "np array of signal time standard deviation (n_chan, n_pix)", unit=u.ns
    )
    relative_gain_mean = Field(
        None, "np array of the relative flat-field coefficient mean (n_chan, n_pix)"
    )
    relative_gain_median = Field(
        None, "np array of the relative flat-field coefficient  median (n_chan, n_pix)"
    )
    relative_gain_std = Field(
        None,
        "np array of the relative flat-field coefficient standard deviation (n_chan, n_pix)",
    )
    relative_time_median = Field(
        None,
        "np array of time (median) - time median averaged over camera (n_chan, n_pix)",
        unit=u.ns,
    )

    charge_median_outliers = Field(
        None, "Boolean np array of charge median outliers (n_chan, n_pix)"
    )
    charge_std_outliers = Field(
        None, "Boolean np array of charge std outliers (n_chan, n_pix)"
    )

    time_median_outliers = Field(
        None, "Boolean np array of pixel time (median) outliers (n_chan, n_pix)"
    )


class PedestalContainer(Container):
    """
    Container for pedestal parameters obtained from a set of
    [n_pedestal] pedestal events
    """

    n_events = Field(-1, "Number of events used for statistics")
    sample_time = Field(
        nan * u.s, "Time associated to the pedestal event set", unit=u.s
    )
    sample_time_min = Field(nan * u.s, "Time of first pedestal event", unit=u.s)
    sample_time_max = Field(nan * u.s, "Time of last pedestal event", unit=u.s)
    charge_mean = Field(None, "np array of pedestal average (n_chan, n_pix)")
    charge_median = Field(None, "np array of the pedestal  median (n_chan, n_pix)")
    charge_std = Field(
        None, "np array of the pedestal standard deviation (n_chan, n_pix)"
    )
    charge_median_outliers = Field(
        None, "Boolean np array of the pedestal median outliers (n_chan, n_pix)"
    )
    charge_std_outliers = Field(
        None, "Boolean np array of the pedestal std outliers (n_chan, n_pix)"
    )


class PixelStatusContainer(Container):
    """
    Container for pixel status information
    It contains masks obtained by several data analysis steps
    At r0/r1 level only the hardware_mask is initialized
    """

    hardware_failing_pixels = Field(
        None,
        "Boolean np array (True = failing pixel) from the hardware pixel status data ("
        "n_chan, n_pix)",
    )

    pedestal_failing_pixels = Field(
        None,
        "Boolean np array (True = failing pixel) from the pedestal data analysis ("
        "n_chan, n_pix)",
    )

    flatfield_failing_pixels = Field(
        None,
        "Boolean np array (True = failing pixel) from the flat-field data analysis ("
        "n_chan, n_pix)",
    )


class WaveformCalibrationContainer(Container):
    """
    Container for the pixel calibration coefficients
    """

    time = Field(nan * u.s, "Time associated to the calibration event", unit=u.s)
    time_min = Field(
        nan * u.s, "Earliest time of validity for the calibration event", unit=u.s
    )
    time_max = Field(
        nan * u.s, "Latest time of validity for the calibration event", unit=u.s
    )

    dc_to_pe = Field(
        None,
        "np array of (digital count) to (photon electron) coefficients (n_chan, n_pix)",
    )

    pedestal_per_sample = Field(
        None,
        "np array of average pedestal value per sample (digital count) (n_chan, n_pix)",
    )

    time_correction = Field(None, "np array of time correction values (n_chan, n_pix)")

    n_pe = Field(
        None, "np array of photo-electrons in calibration signal (n_chan, n_pix)"
    )

    unusable_pixels = Field(
        None,
        "Boolean np array of final calibration data analysis, True = failing pixels (n_chan, n_pix)",
    )


class MonitoringCameraContainer(Container):
    """
    Container for camera monitoring data
    """

    flatfield = Field(
        default_factory=FlatFieldContainer,
        description="Data from flat-field event distributions",
    )
    pedestal = Field(
        default_factory=PedestalContainer,
        description="Data from pedestal event distributions",
    )
    pixel_status = Field(
        default_factory=PixelStatusContainer,
        description="Container for masks with pixel status",
    )
    calibration = Field(
        default_factory=WaveformCalibrationContainer,
        description="Container for calibration coefficients",
    )


class MonitoringContainer(Container):
    """
    Root container for monitoring data (MON)
    """

    # create the camera container
    tel = Field(
        default_factory=partial(Map, MonitoringCameraContainer),
        description="map of tel_id to MonitoringCameraContainer",
    )


class SimulatedShowerDistribution(Container):
    """
    2D histogram of simulated number of showers simulated as function of energy and
    core distance.
    """

    default_prefix = ""

    obs_id = obs_id_field()
    hist_id = Field(-1, description="Histogram ID")
    n_entries = Field(-1, description="Number of entries in the histogram")
    bins_energy = Field(
        None,
        description="array of energy bin lower edges, as in np.histogram",
        unit=u.TeV,
    )
    bins_core_dist = Field(
        None,
        description="array of core-distance bin lower edges, as in np.histogram",
        unit=u.m,
    )
    histogram = Field(
        None, description="array of histogram entries, size (n_bins_x, n_bins_y)"
    )


class ArrayEventContainer(Container):
    """Top-level container for all event information"""

    index = Field(
        default_factory=EventIndexContainer, description="event indexing information"
    )
    r0 = Field(default_factory=R0Container, description="Raw Data")
    r1 = Field(default_factory=R1Container, description="R1 Calibrated Data")
    dl0 = Field(
        default_factory=DL0Container, description="DL0 Data Volume Reduced Data"
    )
    dl1 = Field(default_factory=DL1Container, description="DL1 Calibrated image")
    dl2 = Field(default_factory=DL2Container, description="DL2 reconstruction info")
    simulation = Field(
        None, description="Simulated Event Information", type=SimulatedEventContainer
    )
    trigger = Field(
        default_factory=TriggerContainer, description="central trigger information"
    )
    count = Field(0, description="number of events processed")
    pointing = Field(
        default_factory=PointingContainer,
        description="Array and telescope pointing positions",
    )
    calibration = Field(
        default_factory=EventCalibrationContainer,
        description="Container for calibration coefficients for the current event",
    )
    mon = Field(
        default_factory=MonitoringContainer,
        description="container for event-wise monitoring data (MON)",
    )
    muon = Field(
        default_factory=MuonContainer, description="Container for muon analysis results"
    )


class SchedulingBlockContainer(Container):
    """Stores information about the scheduling block.

    This is a simplified version of the SB model, only storing what is necessary for analysis.
    From :cite:p:`cta-sb-ob-data-model`.
    """

    default_prefix = ""

    sb_id = Field(UNKNOWN_ID, "Scheduling block ID", type=np.uint64)
    sb_type = Field(
        SchedulingBlockType.UNKNOWN,
        description="Type of scheduling block",
        type=SchedulingBlockType,
    )
    producer_id = Field(
        "unknown",
        "Origin of the sb_id, i.e. name of the telescope site or 'simulation'",
        type=str,
    )
    observing_mode = Field(
        ObservingMode.UNKNOWN,
        "Defines how observations within the Scheduling Block are distributed in space",
        type=ObservingMode,
    )
    pointing_mode = Field(
        PointingMode.UNKNOWN, "Defines how the telescope drives move", type=PointingMode
    )


class ObservationBlockContainer(Container):
    """Stores information about the observation"""

    default_prefix = ""
    obs_id = obs_id_field()
    sb_id = Field(UNKNOWN_ID, "ID of the parent SchedulingBlock", type=np.uint64)
    producer_id = Field(
        "unknown",
        "Origin of the obs_id, i.e. name of the telescope site or 'simulation'",
        type=str,
    )

    state = Field(
        ObservationBlockState.UNKNOWN, "State of this OB", type=ObservationBlockState
    )

    subarray_pointing_lat = Field(
        nan * u.deg,
        "latitude of the nominal center coordinate of this observation",
        unit=u.deg,
    )

    subarray_pointing_lon = Field(
        nan * u.deg,
        "longitude of the nominal center coordinate of this observation",
        unit=u.deg,
    )

    subarray_pointing_frame = Field(
        CoordinateFrameType.UNKNOWN,
        (
            "Frame in which the subarray_target is non-moving. If the frame is ALTAZ, "
            "the meaning of (lon,lat) is (azimuth, altitude) while for ICRS it is "
            "(right-ascension, declination)"
        ),
        type=CoordinateFrameType,
    )

    scheduled_duration = Field(
        nan * u.min, "expected duration from scheduler", unit=u.min
    )
    scheduled_start_time = Field(NAN_TIME, "expected start time from scheduler")
    actual_start_time = Field(NAN_TIME, "true start time")
    actual_duration = Field(nan * u.min, "true duration", unit=u.min)
