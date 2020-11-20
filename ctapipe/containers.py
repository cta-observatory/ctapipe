"""
Container structures for data that should be read or written to disk
"""
import enum

from astropy import units as u
from astropy.time import Time
from numpy import nan
import numpy as np

from .core import Container, Field, Map

__all__ = [
    "ArrayEventContainer",
    "ConcentrationContainer",
    "DL0CameraContainer",
    "DL0Container",
    "DL1CameraCalibrationContainer",
    "DL1CameraContainer",
    "DL1Container",
    "EventCalibrationContainer",
    "EventCameraCalibrationContainer",
    "EventIndexContainer",
    "EventType",
    "FlatFieldContainer",
    "HillasParametersContainer",
    "ImageParametersContainer",
    "LeakageContainer",
    "MonitoringCameraContainer",
    "MonitoringContainer",
    "MorphologyContainer",
    "ParticleClassificationContainer",
    "PedestalContainer",
    "PixelStatusContainer",
    "R0CameraContainer",
    "R0Container",
    "R1CameraContainer",
    "R1Container",
    "ReconstructedContainer",
    "ReconstructedEnergyContainer",
    "ReconstructedShowerContainer",
    "SimulatedCameraContainer",
    "SimulatedShowerContainer",
    "SimulatedShowerDistribution",
    "SimulationConfigContainer",
    "TelEventIndexContainer",
    "TimingParametersContainer",
    "TriggerContainer",
    "WaveformCalibrationContainer",
]


# see https://github.com/astropy/astropy/issues/6509
NAN_TIME = Time(np.ma.masked_array(nan, mask=True), format="mjd")


class EventType(enum.Enum):
    """These numbers come from  the document *CTA R1/Event Data Model Specification*
    version 1 revision C.  They may be updated in future revisions"""

    # calibrations are 0-15
    FLATFIELD = 0
    SINGLE_PE = 1
    SKY_PEDESTAL = 2
    DARK_PEDESTAL = 3
    ELECTRONIC_PEDESTAL = 4
    OTHER_CALIBRATION = 15

    # For mono-telescope triggers (not used in MC)
    MUON = 16
    HARDWARE_STEREO = 17

    # ACADA (DAQ) software trigger
    DAQ = 24

    # Standard Physics  stereo trigger
    SUBARRAY = 32

    UNKNOWN = 255


class EventIndexContainer(Container):
    """ index columns to include in event lists, common to all data levels"""

    container_prefix = ""  # don't want to prefix these
    obs_id = Field(0, "observation identifier")
    event_id = Field(0, "event identifier")


class TelEventIndexContainer(Container):
    """
    index columns to include in telescope-wise event lists, common to all data
    levels that have telescope-wise information
    """

    container_prefix = ""  # don't want to prefix these
    obs_id = Field(0, "observation identifier")
    event_id = Field(0, "event identifier")
    tel_id = Field(0, "telescope identifier")


class HillasParametersContainer(Container):
    container_prefix = "hillas"

    intensity = Field(nan, "total intensity (size)")

    x = Field(nan * u.m, "centroid x coordinate", unit=u.m)
    y = Field(nan * u.m, "centroid x coordinate", unit=u.m)
    r = Field(nan * u.m, "radial coordinate of centroid", unit=u.m)
    phi = Field(nan * u.deg, "polar coordinate of centroid", unit=u.deg)

    length = Field(nan * u.m, "standard deviation along the major-axis", unit=u.m)
    length_uncertainty = Field(nan * u.m, "uncertainty of length", unit=u.m)
    width = Field(nan * u.m, "standard spread along the minor-axis", unit=u.m)
    width_uncertainty = Field(nan * u.m, "uncertainty of width", unit=u.m)
    psi = Field(nan * u.deg, "rotation angle of ellipse", unit=u.deg)

    skewness = Field(nan, "measure of the asymmetry")
    kurtosis = Field(nan, "measure of the tailedness")


class LeakageContainer(Container):
    """
    Fraction of signal in 1 or 2-pixel width border from the edge of the
    camera, measured in number of signal pixels or in intensity.
    """

    container_prefix = "leakage"

    pixels_width_1 = Field(
        nan, "fraction of pixels after cleaning that are in camera border of width=1"
    )
    pixels_width_2 = Field(
        nan, "fraction of pixels after cleaning that are in camera border of width=2"
    )
    intensity_width_1 = Field(
        nan,
        "Intensity in photo-electrons after cleaning"
        " that are in the camera border of width=1 pixel",
    )
    intensity_width_2 = Field(
        nan,
        "Intensity in photo-electrons after cleaning"
        " that are in the camera border of width=2 pixels",
    )


class ConcentrationContainer(Container):
    """
    Concentrations are ratios between light amount
    in certain areas of the image and the full image.
    """

    container_prefix = "concentration"
    cog = Field(
        nan, "Percentage of photo-electrons in the three pixels closest to the cog"
    )
    core = Field(nan, "Percentage of photo-electrons inside the hillas ellipse")
    pixel = Field(nan, "Percentage of photo-electrons in the brightest pixel")


class TimingParametersContainer(Container):
    """
    Slope and Intercept of a linear regression of the arrival times
    along the shower main axis
    """

    container_prefix = "timing"
    slope = Field(
        nan / u.m, "Slope of arrival times along main shower axis", unit=1 / u.m
    )
    intercept = Field(nan, "intercept of arrival times along main shower axis")
    deviation = Field(
        nan,
        "Root-mean-square deviation of the pulse times "
        "with respect to the predicted time",
    )


class MorphologyContainer(Container):
    """ Parameters related to pixels surviving image cleaning """

    num_pixels = Field(-1, "Number of usable pixels")
    num_islands = Field(-1, "Number of distinct islands in the image")
    num_small_islands = Field(-1, "Number of <= 2 pixel islands")
    num_medium_islands = Field(-1, "Number of 2-50 pixel islands")
    num_large_islands = Field(-1, "Number of > 50 pixel islands")


class StatisticsContainer(Container):
    """Store descriptive statistics"""

    max = Field(nan, "value of pixel with maximum intensity")
    min = Field(nan, "value of pixel with minimum intensity")
    mean = Field(nan, "mean intensity")
    std = Field(nan, "standard deviation of intensity")
    skewness = Field(nan, "skewness of intensity")
    kurtosis = Field(nan, "kurtosis of intensity")


class IntensityStatisticsContainer(StatisticsContainer):
    container_prefix = "intensity"


class PeakTimeStatisticsContainer(StatisticsContainer):
    container_prefix = "peak_time"


class ImageParametersContainer(Container):
    """ Collection of image parameters """

    container_prefix = "params"
    hillas = Field(HillasParametersContainer(), "Hillas Parameters")
    timing = Field(TimingParametersContainer(), "Timing Parameters")
    leakage = Field(LeakageContainer(), "Leakage Parameters")
    concentration = Field(ConcentrationContainer(), "Concentration Parameters")
    morphology = Field(MorphologyContainer(), "Image Morphology Parameters")
    intensity_statistics = Field(
        IntensityStatisticsContainer(), "Intensity image statistics"
    )
    peak_time_statistics = Field(
        PeakTimeStatisticsContainer(), "Peak time image statistics"
    )


class DL1CameraContainer(Container):
    """
    Storage of output of camera calibration e.g the final calibrated
    image in intensity units and the pulse time.
    """

    image = Field(
        None,
        "Numpy array of camera image, after waveform extraction." "Shape: (n_pixel)",
        dtype=np.float32,
        ndim=1,
    )
    peak_time = Field(
        None,
        "Numpy array containing position of the peak of the pulse as determined by "
        "the extractor. Shape: (n_pixel)",
        dtype=np.float32,
        ndim=1,
    )

    image_mask = Field(
        None,
        "Boolean numpy array where True means the pixel has passed cleaning. Shape: ("
        "n_pixel)",
        dtype=np.bool,
        ndim=1,
    )

    parameters = Field(ImageParametersContainer(), "Parameters derived from images")


class DL1Container(Container):
    """ DL1 Calibrated Camera Images and associated data"""

    tel = Field(Map(DL1CameraContainer), "map of tel_id to DL1CameraContainer")


class DL1CameraCalibrationContainer(Container):
    """
    Storage of DL1 calibration parameters for the current event
    """

    pedestal_offset = Field(
        None,
        "Residual mean pedestal of the waveforms for each pixel."
        " This value is subtracted from the waveforms of each pixel before"
        " the pulse extraction.",
    )
    absolute_factor = Field(
        1,
        "Multiplicative coefficients for the absolute calibration of extracted charge into "
        "physical units (e.g. photoelectrons or photons) for each pixel",
    )
    relative_factor = Field(
        1,
        "Multiplicative Coefficients for the relative correction between pixels to achieve a "
        "uniform charge response (post absolute calibration) from a "
        "uniform illumination.",
    )
    time_shift = Field(
        None,
        "Additive coefficients for the timing correction before charge extraction "
        "for each pixel",
    )


class R0CameraContainer(Container):
    """
    Storage of raw data from a single telescope
    """

    waveform = Field(
        None, ("numpy array containing ADC samples" "(n_channels, n_pixels, n_samples)")
    )


class R0Container(Container):
    """
    Storage of a Merged Raw Data Event
    """

    tel = Field(Map(R0CameraContainer), "map of tel_id to R0CameraContainer")


class R1CameraContainer(Container):
    """
    Storage of r1 calibrated data from a single telescope
    """

    waveform = Field(
        None,
        (
            "numpy array containing a set of images, one per ADC sample"
            "Shape: (n_pixels, n_samples)"
        ),
    )
    selected_gain_channel = Field(
        None,
        (
            "Numpy array containing the gain channel chosen for each pixel. "
            "Shape: (n_pixels)"
        ),
    )


class R1Container(Container):
    """
    Storage of a r1 calibrated Data Event
    """

    tel = Field(Map(R1CameraContainer), "map of tel_id to R1CameraContainer")


class DL0CameraContainer(Container):
    """
    Storage of data volume reduced dl0 data from a single telescope
    """

    waveform = Field(
        None,
        (
            "numpy array containing data volume reduced "
            "p.e. samples"
            "(n_pixels, n_samples). Note this may be a masked array, "
            "if pixels or time slices are zero-suppressed"
        ),
    )

    selected_gain_channel = Field(
        None,
        (
            "Numpy array containing the gain channel chosen for each pixel. "
            "Shape: (n_pixels)"
        ),
    )


class DL0Container(Container):
    """
    Storage of a data volume reduced Event
    """

    tel = Field(Map(DL0CameraContainer), "map of tel_id to DL0CameraContainer")


class SimulatedShowerContainer(Container):
    container_prefix = "true"
    energy = Field(nan * u.TeV, "Simulated Energy", unit=u.TeV)
    alt = Field(nan * u.deg, "Simulated altitude", unit=u.deg)
    az = Field(nan * u.deg, "Simulated azimuth", unit=u.deg)
    core_x = Field(nan * u.m, "Simulated core position (x)", unit=u.m)
    core_y = Field(nan * u.m, "Simulated core position (y)", unit=u.m)
    h_first_int = Field(nan * u.m, "Height of first interaction", unit=u.m)
    x_max = Field(
        nan * u.g / (u.cm ** 2), "Simulated Xmax value", unit=u.g / (u.cm ** 2)
    )
    shower_primary_id = Field(
        -1,
        "Simulated shower primary ID 0 (gamma), 1(e-),"
        "2(mu-), 100*A+Z for nucleons and nuclei,"
        "negative for antimatter.",
    )


class SimulatedCameraContainer(Container):
    """
    True images and parameters derived from them, analgous to the `DL1CameraContainer`
    but for simulated data.
    """

    container_prefix = ""

    true_image = Field(
        None,
        "Numpy array of camera image in PE as simulated before noise has been added. "
        "Shape: (n_pixel)",
        dtype=np.float32,
        ndim=1,
    )

    true_parameters = Field(
        ImageParametersContainer(), "Parameters derived from the true_image"
    )


class SimulatedEventContainer(Container):
    shower = Field(SimulatedShowerContainer(), "True event information")
    tel = Field(Map(SimulatedCameraContainer))


class SimulationConfigContainer(Container):
    """
    Configuration parameters of the simulation
    """

    corsika_version = Field(nan, "CORSIKA version * 1000")
    simtel_version = Field(nan, "sim_telarray version * 1000")
    energy_range_min = Field(
        nan * u.TeV, "Lower limit of energy range of primary particle", unit=u.TeV
    )
    energy_range_max = Field(
        nan * u.TeV, "Upper limit of energy range of primary particle", unit=u.TeV
    )
    prod_site_B_total = Field(nan * u.uT, "total geomagnetic field", unit=u.uT)
    prod_site_B_declination = Field(nan * u.rad, "magnetic declination", unit=u.rad)
    prod_site_B_inclination = Field(nan * u.rad, "magnetic inclination", unit=u.rad)
    prod_site_alt = Field(nan * u.m, "height of observation level", unit=u.m)
    spectral_index = Field(nan, "Power-law spectral index of spectrum")
    shower_prog_start = Field(
        nan, "Time when shower simulation started, CORSIKA: only date"
    )
    shower_prog_id = Field(nan, "CORSIKA=1, ALTAI=2, KASCADE=3, MOCCA=4")
    detector_prog_start = Field(nan, "Time when detector simulation started")
    detector_prog_id = Field(nan, "simtelarray=1")
    num_showers = Field(nan, "Number of showers simulated")
    shower_reuse = Field(nan, "Numbers of uses of each shower")
    max_alt = Field(nan * u.rad, "Maximimum shower altitude", unit=u.rad)
    min_alt = Field(nan * u.rad, "Minimum shower altitude", unit=u.rad)
    max_az = Field(nan * u.rad, "Maximum shower azimuth", unit=u.rad)
    min_az = Field(nan * u.rad, "Minimum shower azimuth", unit=u.rad)
    diffuse = Field(False, "Diffuse Mode On/Off")
    max_viewcone_radius = Field(nan * u.deg, "Maximum viewcone radius", unit=u.deg)
    min_viewcone_radius = Field(nan * u.deg, "Minimum viewcone radius", unit=u.deg)
    max_scatter_range = Field(nan * u.m, "Maximum scatter range", unit=u.m)
    min_scatter_range = Field(nan * u.m, "Minimum scatter range", unit=u.m)
    core_pos_mode = Field(nan, "Core Position Mode (0=Circular, 1=Rectangular)")
    injection_height = Field(nan * u.m, "Height of particle injection", unit=u.m)
    atmosphere = Field(nan * u.m, "Atmospheric model number")
    corsika_iact_options = Field(nan, "CORSIKA simulation options for IACTs")
    corsika_low_E_model = Field(nan, "CORSIKA low-energy simulation physics model")
    corsika_high_E_model = Field(
        nan,
        "CORSIKA physics model ID for high energies "
        "(1=VENUS, 2=SIBYLL, 3=QGSJET, 4=DPMJET, 5=NeXus, 6=EPOS) ",
    )
    corsika_bunchsize = Field(nan, "Number of Cherenkov photons per bunch")
    corsika_wlen_min = Field(
        nan * u.m, "Minimum wavelength of cherenkov light", unit=u.nm
    )
    corsika_wlen_max = Field(
        nan * u.m, "Maximum wavelength of cherenkov light", unit=u.nm
    )
    corsika_low_E_detail = Field(
        nan, "More details on low E interaction model (version etc.)"
    )
    corsika_high_E_detail = Field(
        nan, "More details on high E interaction model (version etc.)"
    )


class TelescopeTriggerContainer(Container):
    container_prefix = ""
    time = Field(NAN_TIME, "Telescope trigger time")
    n_trigger_pixels = Field(-1, "Number of trigger groups (sectors) listed")
    trigger_pixels = Field(None, "pixels involved in the camera trigger")


class TriggerContainer(Container):
    container_prefix = ""
    time = Field(NAN_TIME, "central average time stamp")
    tels_with_trigger = Field(
        [], "List of telescope ids that triggered the array event"
    )
    event_type = Field(EventType.SUBARRAY, "Event type")
    tel = Field(Map(TelescopeTriggerContainer), "telescope-wise trigger information")


class ReconstructedShowerContainer(Container):
    """
    Standard output of algorithms reconstructing shower geometry
    """

    alt = Field(nan * u.deg, "reconstructed altitude", unit=u.deg)
    alt_uncert = Field(nan * u.deg, "reconstructed altitude uncertainty", unit=u.deg)
    az = Field(nan * u.deg, "reconstructed azimuth", unit=u.deg)
    az_uncert = Field(nan * u.deg, "reconstructed azimuth uncertainty", unit=u.deg)
    core_x = Field(
        nan * u.m, "reconstructed x coordinate of the core position", unit=u.m
    )
    core_y = Field(
        nan * u.m, "reconstructed y coordinate of the core position", unit=u.m
    )
    core_uncert = Field(
        nan * u.m, "uncertainty of the reconstructed core position", unit=u.m
    )
    h_max = Field(nan * u.m, "reconstructed height of the shower maximum", unit=u.m)
    h_max_uncert = Field(nan * u.m, "uncertainty of h_max", unit=u.m)
    is_valid = Field(
        False,
        (
            "direction validity flag. True if the shower direction"
            "was properly reconstructed by the algorithm"
        ),
    )
    tel_ids = Field(
        [], ("list of the telescope ids used in the" " reconstruction of the shower")
    )
    average_intensity = Field(
        nan, "average intensity of the intensities used for reconstruction"
    )
    goodness_of_fit = Field(nan, "measure of algorithm success (if fit)")


class ReconstructedEnergyContainer(Container):
    """
    Standard output of algorithms estimating energy
    """

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
    tel_ids = Field(
        [],
        (
            "array containing the telescope ids used in the"
            " reconstruction of the shower"
        ),
    )
    goodness_of_fit = Field(0.0, "goodness of the algorithm fit")


class ParticleClassificationContainer(Container):
    """
    Standard output of gamma/hadron classification algorithms
    """

    # TODO: Do people agree on this? This is very MAGIC-like.
    # TODO: Perhaps an integer classification to support different classes?
    # TODO: include an error on the prediction?
    prediction = Field(
        0.0,
        (
            "prediction of the classifier, defined between "
            "[0,1], where values close to 0 are more "
            "gamma-like, and values close to 1 more "
            "hadron-like"
        ),
    )
    is_valid = Field(
        False,
        (
            "classificator validity flag. True if the "
            "predition was successful within the algorithm "
            "validity range"
        ),
    )

    # TODO: KPK: is this different than the list in the reco
    # container? Why repeat?
    tel_ids = Field(
        [],
        (
            "array containing the telescope ids used "
            "in the reconstruction of the shower"
        ),
    )
    goodness_of_fit = Field(0.0, "goodness of the algorithm fit")


class ReconstructedContainer(Container):
    """ collect reconstructed shower info from multiple algorithms """

    shower = Field(
        Map(ReconstructedShowerContainer), "Map of algorithm name to shower info"
    )
    energy = Field(
        Map(ReconstructedEnergyContainer), "Map of algorithm name to energy info"
    )
    classification = Field(
        Map(ParticleClassificationContainer),
        "Map of algorithm name to classification info",
    )


class TelescopePointingContainer(Container):
    """
    Container holding pointing information for a single telescope
    after all necessary correction and calibration steps.
    These values should be used in the reconstruction to transform
    between camera and sky coordinates.
    """

    azimuth = Field(nan * u.rad, "Azimuth, measured N->E", unit=u.rad)
    altitude = Field(nan * u.rad, "Altitude", unit=u.rad)


class PointingContainer(Container):
    tel = Field(Map(TelescopePointingContainer), "Telescope pointing positions")
    array_azimuth = Field(nan * u.rad, "Array pointing azimuth", unit=u.rad)
    array_altitude = Field(nan * u.rad, "Array pointing altitude", unit=u.rad)
    array_ra = Field(nan * u.rad, "Array pointing right ascension", unit=u.rad)
    array_dec = Field(nan * u.rad, "Array pointing declination", unit=u.rad)


class EventCameraCalibrationContainer(Container):
    """
    Container for the calibration coefficients for the current event and camera
    """

    dl1 = Field(
        DL1CameraCalibrationContainer(), "Container for DL1 calibration coefficients"
    )


class EventCalibrationContainer(Container):
    """
    Container for calibration coefficients for the current event
    """

    # create the camera container
    tel = Field(
        Map(EventCameraCalibrationContainer),
        "map of tel_id to EventCameraCalibrationContainer",
    )


class MuonRingContainer(Container):
    """Container for the result of a ring fit, center_x, center_y"""

    center_x = Field(nan * u.deg, "center (x) of the fitted muon ring", unit=u.deg)
    center_y = Field(nan * u.deg, "center (y) of the fitted muon ring", unit=u.deg)
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
        nan, "MSE of the deviation of all pixels after cleaning from the ring fit"
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

    flatfield = Field(FlatFieldContainer(), "Data from flat-field event distributions")
    pedestal = Field(PedestalContainer(), "Data from pedestal event distributions")
    pixel_status = Field(
        PixelStatusContainer(), "Container for masks with pixel status"
    )
    calibration = Field(
        WaveformCalibrationContainer(), "Container for calibration coefficients"
    )


class MonitoringContainer(Container):
    """
    Root container for monitoring data (MON)
    """

    # create the camera container
    tel = Field(
        Map(MonitoringCameraContainer), "map of tel_id to MonitoringCameraContainer"
    )


class SimulatedShowerDistribution(Container):
    """
    2D histogram of simulated number of showers simulated as function of energy and
    core distance.
    """

    container_prefix = ""

    obs_id = Field(-1, "links to which events this corresponds to")
    hist_id = Field(-1, "Histogram ID")
    num_entries = Field(-1, "Number of entries in the histogram")
    bins_energy = Field(
        None, "array of energy bin lower edges, as in np.histogram", unit=u.TeV
    )
    bins_core_dist = Field(
        None, "array of core-distance bin lower edges, as in np.histogram", unit=u.m
    )
    histogram = Field(None, "array of histogram entries, size (n_bins_x, n_bins_y)")


class ArrayEventContainer(Container):
    """ Top-level container for all event information """

    index = Field(EventIndexContainer(), "event indexing information")
    r0 = Field(R0Container(), "Raw Data")
    r1 = Field(R1Container(), "R1 Calibrated Data")
    dl0 = Field(DL0Container(), "DL0 Data Volume Reduced Data")
    dl1 = Field(DL1Container(), "DL1 Calibrated image")
    dl2 = Field(ReconstructedContainer(), "Reconstructed Shower Information")
    simulation = Field(SimulatedEventContainer(), "Simulated Event Information")
    trigger = Field(TriggerContainer(), "central trigger information")
    count = Field(0, "number of events processed")
    pointing = Field(PointingContainer(), "Array and telescope pointing positions")
    calibration = Field(
        EventCalibrationContainer(),
        "Container for calibration coefficients for the current event",
    )
    mon = Field(MonitoringContainer(), "container for event-wise monitoring data (MON)")
