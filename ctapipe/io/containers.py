"""
Container structures for data that should be read or written to disk
"""

import numpy as np
from astropy import units as u
from astropy.time import Time
from numpy import nan

from ..core import Container, Field, DeprecatedField, Map
from ..instrument import SubarrayDescription

__all__ = [
    "InstrumentContainer",
    "R0Container",
    "R0CameraContainer",
    "R1Container",
    "R1CameraContainer",
    "DL0Container",
    "DL0CameraContainer",
    "DL1Container",
    "DL1CameraContainer",
    "EventCameraCalibrationContainer",
    "EventCalibrationContainer",
    "SST1MContainer",
    "SST1MCameraContainer",
    "MCEventContainer",
    "MCHeaderContainer",
    "MCCameraEventContainer",
    "DL1CameraCalibrationContainer",
    "CentralTriggerContainer",
    "ReconstructedContainer",
    "ReconstructedShowerContainer",
    "ReconstructedEnergyContainer",
    "ParticleClassificationContainer",
    "DataContainer",
    "SST1MDataContainer",
    "HillasParametersContainer",
    "LeakageContainer",
    "ConcentrationContainer",
    "MorphologyContainer",
    "TimingParametersContainer",
    "FlatFieldContainer",
    "PedestalContainer",
    "PixelStatusContainer",
    "WaveformCalibrationContainer",
    "MonitoringCameraContainer",
    "MonitoringContainer",
    "EventAndMonDataContainer",
    "EventIndexContainer",
    "TelEventIndexContainer",
    "ImageParametersContainer",
    "SimulatedShowerDistribution",
]


class EventIndexContainer(Container):
    """ index columns to include in event lists, common to all data levels"""

    container_prefix = ""  # don't want to prefix these

    event_id = Field(0, "event identifier")
    obs_id = Field(0, "observation identifier")


class TelEventIndexContainer(EventIndexContainer):
    """
    index columns to include in telescope-wise event lists, common to all data
    levels that have telescope-wise information
    """

    container_prefix = ""  # don't want to prefix these

    tel_id = Field(0, "telescope identifier")
    tel_type_id = Field(0, "telescope type id number (integer)")


class SST1MCameraContainer(Container):
    pixel_flags = Field(None, "numpy array containing pixel flags")
    digicam_baseline = Field(None, "Baseline computed by DigiCam")
    local_camera_clock = Field(float, "camera timestamp")
    gps_time = Field(float, "gps timestamp")
    camera_event_type = Field(int, "camera event type")
    array_event_type = Field(int, "array event type")
    trigger_input_traces = Field(None, "trigger patch trace (n_patches)")
    trigger_output_patch7 = Field(None, "trigger 7 patch cluster trace (n_clusters)")
    trigger_output_patch19 = Field(None, "trigger 19 patch cluster trace (n_clusters)")


class SST1MContainer(Container):
    tels_with_data = Field([], "list of telescopes with data")
    tel = Field(Map(SST1MCameraContainer), "map of tel_id to SST1MCameraContainer")


# todo: change some of these Maps to be just 3D NDarrays?


class InstrumentContainer(Container):
    """Storage of header info that does not change with event. This is a
    temporary hack until the Instrument module and database is fully
    implemented.  Eventually static information like this will not be
    part of the data stream, but be loaded and accessed from
    functions.

    """

    subarray = Field(
        SubarrayDescription("MonteCarloArray"),
        "SubarrayDescription from the instrument module",
    )


class DL1CameraContainer(Container):
    """
    Storage of output of camera calibration e.g the final calibrated
    image in intensity units and the pulse time.
    """

    image = Field(
        None,
        "Numpy array of camera image, after waveform extraction." "Shape: (n_pixel)",
    )
    pulse_time = Field(
        None,
        "Numpy array containing position of the pulse as determined by "
        "the extractor."
        "Shape: (n_pixel, n_samples)",
    )


class DL1Container(Container):
    """ DL1 Calibrated Camera Images and associated data"""

    tel = Field(Map(DL1CameraContainer), "map of tel_id to DL1CameraContainer")


class DL1CameraCalibrationContainer(Container):
    """
    Storage of DL1 calibration parameters for the current event
    """

    pedestal_offset = Field(
        0,
        "Additive coefficients for the pedestal calibration of extracted charge "
        "for each pixel"
    )
    absolute_factor = Field(
        1,
        "Multiplicative coefficients for the absolute calibration of extracted charge into "
        "physical units (e.g. photoelectrons or photons) for each pixel"
    )
    relative_factor = Field(
        1,
        "Multiplicative Coefficients for the relative correction between pixels to achieve a "
        "uniform charge response (post absolute calibration) from a "
        "uniform illumination."
    )
    time_shift = Field(
        0,
        "Additive coefficients for the timing correction before charge extraction "
        "for each pixel"
    )


class R0CameraContainer(Container):
    """
    Storage of raw data from a single telescope
    """

    trigger_time = Field(
        None, "Telescope trigger time, start of waveform " "readout, None for MCs"
    )
    trigger_type = Field(0o0, "camera's event trigger type if applicable")
    num_trig_pix = Field(0, "Number of trigger groups (sectors) listed")
    trig_pix_id = Field(None, "pixels involved in the camera trigger")
    waveform = Field(
        None, ("numpy array containing ADC samples" "(n_channels, n_pixels, n_samples)")
    )


class R0Container(Container):
    """
    Storage of a Merged Raw Data Event
    """

    obs_id = DeprecatedField(-1, "observation ID", reason="moved to event.index")
    event_id = DeprecatedField(-1, "event id number", reason="moved to event.index")
    tels_with_data = Field([], "list of telescopes with data")
    tel = Field(Map(R0CameraContainer), "map of tel_id to R0CameraContainer")


class R1CameraContainer(Container):
    """
    Storage of r1 calibrated data from a single telescope
    """

    trigger_time = Field(None, "Telescope trigger time, start of waveform " "readout")
    trigger_type = Field(0o0, "camera trigger type")

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

    obs_id = DeprecatedField(-1, "observation ID", reason="moved to event.index")
    event_id = DeprecatedField(-1, "event id number", reason="moved to event.index")
    tels_with_data = Field([], "list of telescopes with data")
    tel = Field(Map(R1CameraContainer), "map of tel_id to R1CameraContainer")


class DL0CameraContainer(Container):
    """
    Storage of data volume reduced dl0 data from a single telescope
    """

    trigger_time = Field(None, "Telescope trigger time, start of waveform " "readout")
    trigger_type = Field(0o0, "camera trigger type")

    waveform = Field(
        None,
        (
            "numpy array containing data volume reduced "
            "p.e. samples"
            "(n_pixels, n_samples). Note this may be a masked array, "
            "if pixels or time slices are zero-suppressed"
        ),
    )


class DL0Container(Container):
    """
    Storage of a data volume reduced Event
    """

    obs_id = DeprecatedField(
        -1, "observation ID", reason="moved to event.index"
    )  # use event.index.obs_id
    event_id = DeprecatedField(
        -1, "event id number", reason="moved to event.index"
    )  # use event.index.event_id
    tels_with_data = Field([], "list of telescopes with data")
    tel = Field(Map(DL0CameraContainer), "map of tel_id to DL0CameraContainer")


class MCCameraEventContainer(Container):
    """
    Storage of mc data for a single telescope that change per event
    """

    photo_electron_image = Field(
        Map(), "reference image in pure photoelectrons, with no noise"
    )
    # todo: move to instrument (doesn't change per event)
    reference_pulse_shape = Field(None, "reference pulse shape for each channel")
    # todo: move to instrument or a static MC container (don't change per
    # event)
    time_slice = Field(0, "width of time slice", unit=u.ns)
    dc_to_pe = Field(None, "DC/PE calibration arrays from MC file")
    pedestal = Field(None, "pedestal calibration arrays from MC file")
    azimuth_raw = Field(0, "Raw azimuth angle [radians from N->E] for the telescope")
    altitude_raw = Field(0, "Raw altitude angle [radians] for the telescope")
    azimuth_cor = Field(
        0, "the tracking Azimuth corrected for pointing errors for the telescope"
    )
    altitude_cor = Field(
        0, "the tracking Altitude corrected for pointing errors for the telescope"
    )


class MCEventContainer(Container):
    """
    Monte-Carlo
    """

    energy = Field(0.0, "Monte-Carlo Energy", unit=u.TeV)
    alt = Field(0.0, "Monte-carlo altitude", unit=u.deg)
    az = Field(0.0, "Monte-Carlo azimuth", unit=u.deg)
    core_x = Field(0.0, "MC core position", unit=u.m)
    core_y = Field(0.0, "MC core position", unit=u.m)
    h_first_int = Field(0.0, "Height of first interaction")
    x_max = Field(0.0, "MC Xmax value", unit=u.g / (u.cm ** 2))
    shower_primary_id = Field(
        None,
        "MC shower primary ID 0 (gamma), 1(e-),"
        "2(mu-), 100*A+Z for nucleons and nuclei,"
        "negative for antimatter.",
    )
    tel = Field(Map(MCCameraEventContainer), "map of tel_id to MCCameraEventContainer")


class MCHeaderContainer(Container):
    """
    Monte-Carlo information that doesn't change per event
    """

    run_array_direction = Field(
        [],
        (
            "the tracking/pointing direction in "
            "[radians]. Depending on 'tracking_mode' "
            "this either contains: "
            "[0]=Azimuth, [1]=Altitude in mode 0, "
            "OR "
            "[0]=R.A., [1]=Declination in mode 1."
        ),
    )
    corsika_version = Field(np.nan, "CORSIKA version * 1000")
    simtel_version = Field(np.nan, "sim_telarray version * 1000")
    energy_range_min = Field(
        np.nan, "Lower limit of energy range " "of primary particle", unit=u.TeV
    )
    energy_range_max = Field(
        np.nan, "Upper limit of energy range " "of primary particle", unit=u.TeV
    )
    prod_site_B_total = Field(np.nan, "total geomagnetic field", unit=u.uT)
    prod_site_B_declination = Field(np.nan, "magnetic declination", unit=u.rad)
    prod_site_B_inclination = Field(np.nan, "magnetic inclination", unit=u.rad)
    prod_site_alt = Field(np.nan, "height of observation level", unit=u.m)
    prod_site_array = Field("None", "site array")
    prod_site_coord = Field("None", "site (long., lat.) coordinates")
    prod_site_subarray = Field("None", "site subarray")
    spectral_index = Field(np.nan, "Power-law spectral index of spectrum")
    shower_prog_start = Field(
        np.nan,
        """Time when shower simulation started,
                              CORSIKA: only date""",
    )
    shower_prog_id = Field(np.nan, "CORSIKA=1, ALTAI=2, KASCADE=3, MOCCA=4")
    detector_prog_start = Field(np.nan, "Time when detector simulation started")
    detector_prog_id = Field(np.nan, "simtelarray=1")
    num_showers = Field(np.nan, "Number of showers simulated")
    shower_reuse = Field(np.nan, "Numbers of uses of each shower")
    max_alt = Field(np.nan, "Maximimum shower altitude", unit=u.rad)
    min_alt = Field(np.nan, "Minimum shower altitude", unit=u.rad)
    max_az = Field(np.nan, "Maximum shower azimuth", unit=u.rad)
    min_az = Field(np.nan, "Minimum shower azimuth", unit=u.rad)
    diffuse = Field(np.nan, "Diffuse Mode On/Off")
    max_viewcone_radius = Field(np.nan, "Maximum viewcone radius", unit=u.deg)
    min_viewcone_radius = Field(np.nan, "Minimum viewcone radius", unit=u.deg)
    max_scatter_range = Field(np.nan, "Maximum scatter range", unit=u.m)
    min_scatter_range = Field(np.nan, "Minimum scatter range", unit=u.m)
    core_pos_mode = Field(np.nan, "Core Position Mode (fixed/circular/...)")
    injection_height = Field(np.nan, "Height of particle injection", unit=u.m)
    atmosphere = Field(np.nan, "Atmospheric model number")
    corsika_iact_options = Field(np.nan, "Detector MC information")
    corsika_low_E_model = Field(np.nan, "Detector MC information")
    corsika_high_E_model = Field(np.nan, "Detector MC information")
    corsika_bunchsize = Field(np.nan, "Number of photons per bunch")
    corsika_wlen_min = Field(np.nan, "Minimum wavelength of cherenkov light", unit=u.nm)
    corsika_wlen_max = Field(np.nan, "Maximum wavelength of cherenkov light", unit=u.nm)
    corsika_low_E_detail = Field(np.nan, "Detector MC information")
    corsika_high_E_detail = Field(np.nan, "Detector MC information")


class CentralTriggerContainer(Container):
    gps_time = Field(Time, "central average time stamp")
    tels_with_trigger = Field([], "list of telescopes with data")


class ReconstructedShowerContainer(Container):
    """
    Standard output of algorithms reconstructing shower geometry
    """

    alt = Field(0.0, "reconstructed altitude", unit=u.deg)
    alt_uncert = Field(0.0, "reconstructed altitude uncertainty", unit=u.deg)
    az = Field(0.0, "reconstructed azimuth", unit=u.deg)
    az_uncert = Field(0.0, "reconstructed azimuth uncertainty", unit=u.deg)
    core_x = Field(0.0, "reconstructed x coordinate of the core position", unit=u.m)
    core_y = Field(0.0, "reconstructed y coordinate of the core position", unit=u.m)
    core_uncert = Field(0.0, "uncertainty of the reconstructed core position", unit=u.m)
    h_max = Field(0.0, "reconstructed height of the shower maximum")
    h_max_uncert = Field(0.0, "uncertainty of h_max")
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
        0.0, "average intensity of the intensities used for reconstruction"
    )
    goodness_of_fit = Field(0.0, "measure of algorithm success (if fit)")


class ReconstructedEnergyContainer(Container):
    """
    Standard output of algorithms estimating energy
    """

    energy = Field(-1.0, "reconstructed energy", unit=u.TeV)
    energy_uncert = Field(-1.0, "reconstructed energy uncertainty", unit=u.TeV)
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

    tels_with_data = Field([], "list of telescopes with data")

    # create the camera container
    tel = Field(
        Map(EventCameraCalibrationContainer),
        "map of tel_id to EventCameraCalibrationContainer"
    )


class DataContainer(Container):
    """ Top-level container for all event information """

    event_type = Field("data", "Event type")
    index = Field(EventIndexContainer(), "event indexing information")
    r0 = Field(R0Container(), "Raw Data")
    r1 = Field(R1Container(), "R1 Calibrated Data")
    dl0 = Field(DL0Container(), "DL0 Data Volume Reduced Data")
    dl1 = Field(DL1Container(), "DL1 Calibrated image")
    dl2 = Field(ReconstructedContainer(), "Reconstructed Shower Information")
    mc = Field(MCEventContainer(), "Monte-Carlo data")
    mcheader = Field(MCHeaderContainer(), "Monte-Carlo run header data")
    trig = Field(CentralTriggerContainer(), "central trigger information")
    count = Field(0, "number of events processed")
    inst = DeprecatedField(
        InstrumentContainer(),
        "instrumental information ",
        reason="will be separated from event structure in future version",
    )
    pointing = Field(Map(TelescopePointingContainer), "Telescope pointing positions")
    calibration = Field(
        EventCalibrationContainer(),
        "Container for calibration coefficients for the current event"
    )


class SST1MDataContainer(DataContainer):
    sst1m = Field(SST1MContainer(), "optional SST1M Specific Information")


class MuonRingParameter(Container):
    """
    Storage of muon ring fit output

    Parameters
    ----------

    obs_id : int
        run number
    event_id : int
        event number
    tel_id : int
        telescope ID
    ring_center_x, ring_center_y, ring_radius, ring_phi, ring_inclination:
        center position, radius, orientation and inlination of the fitted ring
    ring_chi2_fit:
        chi squared of the ring fit
    ring_cov_matrix:
        covariance matrix of ring parameters
    ring_containment:
        angular containment of the ring
    """

    obs_id = Field(0, "run identification number")
    event_id = Field(0, "event identification number")
    tel_id = Field(0, "telescope identification number")
    ring_center_x = Field(0.0, "centre (x) of the fitted muon ring")
    ring_center_y = Field(0.0, "centre (y) of the fitted muon ring")
    ring_radius = Field(0.0, "radius of the fitted muon ring")
    ring_phi = Field(0.0, "Orientation of fitted ring")
    ring_inclination = Field(0.0, "Inclination of fitted ring")
    ring_chi2_fit = Field(0.0, "chisquare of the muon ring fit")
    ring_cov_matrix = Field(0.0, "covariance matrix of the muon ring fit")
    ring_containment = Field(0.0, "containment of the ring inside the camera")
    ring_fit_method = Field("", "fitting method used for the muon ring")
    inputfile = Field("", "input file")


class MuonIntensityParameter(Container):
    """
    Storage of muon intensity fit output

    Parameters
    ----------

    obs_id : int
        run number
    event_id : int
        event number
    tel_id : int
        telescope ID
    impact_parameter: float
        reconstructed impact parameter
    impact_parameter_chi2:
        chi squared impact parameter
    intensity_cov_matrix:
        Covariance matrix of impact parameters or alternatively:
        full 5x5 covariance matrix for the complete fit (ring + impact)
    impact_parameter_pos_x, impact_parameter_pos_y:
        position on the mirror of the muon impact
    COG_x, COG_y:
        center of gravity
    optical_efficiency_muon:
        optical muon efficiency from intensity fit
    ring_completeness:
        completeness of the ring
    ring_pix_completeness:
        pixel completeness of the ring
    ring_num_pixel: int
        Number of pixels composing the ring
    ring_size:
        ring size
    off_ring_size:
        size outside of the ring
    ring_width:
        ring width
    ring_time_width:
        standard deviation of the photons time arrival
    prediction: dict
        ndarray of the predicted charge in all pixels
    mask:
        ndarray of the mask used on the image for fitting

    """

    obs_id = DeprecatedField(
        0, "run identification number", reason="moved to event.index"
    )
    event_id = DeprecatedField(
        0, "event identification number", reason="moved to event.index"
    )
    tel_id = DeprecatedField(
        0, "telescope identification number", reason="moved to event.index"
    )
    ring_completeness = Field(0.0, "fraction of ring present")
    ring_pix_completeness = Field(0.0, "fraction of pixels present in the ring")
    ring_num_pixel = Field(0, "number of pixels in the ring image")
    ring_size = Field(0.0, "size of the ring in pe")
    off_ring_size = Field(0.0, "image size outside of ring in pe")
    ring_width = Field(0.0, "width of the muon ring in degrees")
    ring_time_width = Field(0.0, "duration of the ring image sequence")
    impact_parameter = Field(
        0.0, "distance of muon impact position from centre of mirror"
    )
    impact_parameter_chi2 = Field(0.0, "impact parameter chi squared")
    intensity_cov_matrix = Field(0.0, "covariance matrix of intensity")
    impact_parameter_pos_x = Field(0.0, "impact parameter x position")
    impact_parameter_pos_y = Field(0.0, "impact parameter y position")
    COG_x = Field(0.0, "Centre of Gravity x")
    COG_y = Field(0.0, "Centre of Gravity y")
    prediction = Field([], "image prediction")
    mask = Field([], "image pixel mask")
    optical_efficiency_muon = Field(0.0, "optical efficiency muon")
    intensity_fit_method = Field("", "intensity fit method")
    inputfile = Field("", "input file")


class HillasParametersContainer(Container):
    container_prefix = "hillas"

    intensity = Field(nan, "total intensity (size)")

    x = Field(nan, "centroid x coordinate")
    y = Field(nan, "centroid x coordinate")
    r = Field(nan, "radial coordinate of centroid")
    phi = Field(nan, "polar coordinate of centroid", unit=u.deg)

    length = Field(nan, "RMS spread along the major-axis")
    width = Field(nan, "RMS spread along the minor-axis")
    psi = Field(nan, "rotation angle of ellipse", unit=u.deg)

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
    slope = Field(nan, "Slope of arrival times along main shower axis")
    slope_err = Field(nan, "Uncertainty `slope`")
    intercept = Field(nan, "intercept of arrival times along main shower axis")
    intercept_err = Field(nan, "Uncertainty `intercept`")
    deviation = Field(
        nan,
        "Root-mean-square deviation of the pulse times "
        "with respect to the predicted time",
    )


class MorphologyContainer(Container):
    """ Parameters related to pixels surviving image cleaning """

    num_pixels = Field(np.nan, "Number of usable pixels")
    num_islands = Field(np.nan, "Number of distinct islands in the image")
    num_small_islands = Field(np.nan, "Number of <= 2 pixel islands")
    num_medium_islands = Field(np.nan, "Number of 2-50 pixel islands")
    num_large_islands = Field(np.nan, "Number of > 10 pixel islands")


class ImageParametersContainer(Container):
    """ Collection of image parameters """

    container_prefix = "params"
    hillas = Field(HillasParametersContainer(), "Hillas Parameters")
    timing = Field(TimingParametersContainer(), "Timing Parameters")
    leakage = Field(LeakageContainer(), "Leakage Parameters")
    concentration = Field(ConcentrationContainer(), "Concentration Parameters")
    morphology = Field(MorphologyContainer(), "Morphology Parameters")


class FlatFieldContainer(Container):
    """
    Container for flat-field parameters obtained from a set of
    [n_events] flat-field events
    """

    sample_time = Field(0, "Time associated to the flat-field event set ", unit=u.s)
    sample_time_range = Field(
        [], "Range of time of the flat-field events [t_min, t_max]", unit=u.s
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

    n_events = Field(0, "Number of events used for statistics")
    sample_time = Field(0, "Time associated to the pedestal event set", unit=u.s)
    sample_time_range = Field(
        [], "Range of time of the pedestal events [t_min, t_max]", unit=u.s
    )
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
    time = Field(0, "Time associated to the calibration event", unit=u.s)
    time_range = Field(
        [],
        "Range of time of validity for the calibration event [t_min, t_max]",
        unit=u.s,
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

    tels_with_data = Field([], "list of telescopes with data")

    # create the camera container
    tel = Field(
        Map(MonitoringCameraContainer), "map of tel_id to MonitoringCameraContainer"
    )


class EventAndMonDataContainer(DataContainer):
    """
    Data container including monitoring information
    """

    mon = Field(MonitoringContainer(), "container for monitoring data (MON)")


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
