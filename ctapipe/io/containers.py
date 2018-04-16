"""
Container structures for data that should be read or written to disk
"""

from astropy import units as u
from astropy.time import Time
from numpy import nan
import numpy as np

from ..core import Container, Field, Map
from ..instrument import SubarrayDescription

__all__ = ['InstrumentContainer',
           'R0Container',
           'R0CameraContainer',
           'R1Container',
           'R1CameraContainer',
           'DL0Container',
           'DL0CameraContainer',
           'DL1Container',
           'DL1CameraContainer',
           'MCEventContainer',
           'MCHeaderContainer',
           'MCCameraEventContainer',
           'CameraCalibrationContainer',
           'CentralTriggerContainer',
           'ReconstructedContainer',
           'ReconstructedShowerContainer',
           'ReconstructedEnergyContainer',
           'ParticleClassificationContainer',
           'DataContainer',
           'HillasParametersContainer']


class SST1MCameraContainer(Container):
    pixel_flags = Field(None, 'numpy array containing pixel flags')
    digicam_baseline = Field(None, 'Baseline computed by DigiCam')
    local_camera_clock = Field(float, "camera timestamp")
    gps_time = Field(float, "gps timestamp")
    camera_event_type = Field(int, "camera event type")
    array_event_type = Field(int, "array event type")
    trigger_input_traces = Field(None, "trigger patch trace (n_patches)")
    trigger_output_patch7 = Field(
        None,
        "trigger 7 patch cluster trace (n_clusters)")
    trigger_output_patch19 = Field(
        None,
        "trigger 19 patch cluster trace (n_clusters)")

    def fill_from_zfile_event(self, event, pixel_sort_ids):
        self.pixel_flags = event.pixels_flags[pixel_sort_ids]
        self.digicam_baseline = event.hiGain.waveforms.baselines[
            pixel_sort_ids]
        self.local_camera_clock = (
            event.local_time_sec * 1E9 + event.local_time_nanosec)
        self.gps_time = (
            event.trig.timeSec * 1E9 + event.trig.timeNanoSec)
        self.camera_event_type = event.event_type
        self.array_event_type = event.eventType
        self.trigger_input_traces = event.trigger_input_traces
        self.trigger_output_patch7 = event.trigger_output_patch7
        self.trigger_output_patch19 = event.trigger_output_patch19


class SST1MContainer(Container):
    tels_with_data = Field([], "list of telescopes with data")
    tel = Field(
        Map(SST1MCameraContainer),
        "map of tel_id to SST1MCameraContainer")

    def fill_from_zfile_event(self, event, pixel_sort_ids):
        self.tels_with_data = [event.telescopeID, ]
        sst1m_cam_container = self.tel[event.telescopeID]
        sst1m_cam_container.fill_from_zfile_event(
            event,
            pixel_sort_ids,
        )


# todo: change some of these Maps to be just 3D NDarrays?


class InstrumentContainer(Container):
    """Storage of header info that does not change with event. This is a
    temporary hack until the Instrument module and database is fully
    implemented.  Eventually static information like this will not be
    part of the data stream, but be loaded and accessed from
    functions.

    """

    subarray = Field(SubarrayDescription("MonteCarloArray"),
                     "SubarrayDescription from the instrument module")


class DL1CameraContainer(Container):
    """Storage of output of camera calibration e.g the final calibrated
    image in intensity units and other per-event calculated
    calibration information.
    """
    image = Field(
        None,
        "np array of camera image, after waveform integration (N_pix)"
    )
    gain_channel = Field(None, "boolean numpy array of which gain channel was "
                               "used for each pixel in the image ")
    extracted_samples = Field(
        None,
        "numpy array of bools indicating which samples were included in the "
        "charge extraction as a result of the charge extractor chosen. "
        "Shape=(nchan, npix, nsamples)."
    )
    peakpos = Field(
        None,
        "numpy array containing position of the peak as determined by "
        "the peak-finding algorithm for each pixel"
    )
    cleaned = Field(
        None, "numpy array containing the waveform after cleaning"
    )


class CameraCalibrationContainer(Container):
    """
    Storage of externally calculated calibration parameters (not per-event)
    """
    dc_to_pe = Field(None, "DC/PE calibration arrays from MC file")
    pedestal = Field(None, "pedestal calibration arrays from MC file")


class DL1Container(Container):
    """ DL1 Calibrated Camera Images and associated data"""
    tel = Field(Map(DL1CameraContainer), "map of tel_id to DL1CameraContainer")


class R0CameraContainer(Container):
    """
    Storage of raw data from a single telescope
    """
    trigger_time = Field(None, "Telescope trigger time, start of waveform "
                               "readout, None for MCs")
    trigger_type = Field(0o0, "camera's event trigger type if applicable")
    image = Field(None, (
        "numpy array containing integrated ADC data "
        "(n_channels x n_pixels) DEPRECATED"
    ))  # to be removed, since this doesn't exist in real data and useless in mc
    waveform = Field(None, (
        "numpy array containing ADC samples"
        "(n_channels x n_pixels, n_samples)"
    ))
    num_samples = Field(None, "number of time samples for telescope")


class R0Container(Container):
    """
    Storage of a Merged Raw Data Event
    """

    obs_id = Field(-1, "observation ID")
    event_id = Field(-1, "event id number")
    tels_with_data = Field([], "list of telescopes with data")
    tel = Field(Map(R0CameraContainer), "map of tel_id to R0CameraContainer")


class R1CameraContainer(Container):
    """
    Storage of r1 calibrated data from a single telescope
    """
    trigger_time = Field(None, "Telescope trigger time, start of waveform "
                               "readout")
    trigger_type = Field(0o0, "camera trigger type")

    waveform = Field(None, (
        "numpy array containing a set of images, one per ADC sample"
        "(n_channels x n_pixels, n_samples)"
    ))


class R1Container(Container):
    """
    Storage of a r1 calibrated Data Event
    """

    obs_id = Field(-1, "observation ID")
    event_id = Field(-1, "event id number")
    tels_with_data = Field([], "list of telescopes with data")
    tel = Field(Map(R1CameraContainer), "map of tel_id to R1CameraContainer")


class DL0CameraContainer(Container):
    """
    Storage of data volume reduced dl0 data from a single telescope
    """
    trigger_time = Field(None, "Telescope trigger time, start of waveform "
                               "readout")
    trigger_type = Field(0o0, "camera trigger type")

    waveform = Field(None, (
        "numpy array containing data volume reduced "
        "p.e. samples"
        "(n_pixels, n_samples). Note this may be a masked array, "
        "if pixels or time slices are zero-suppressed"
    ))


class DL0Container(Container):
    """
    Storage of a data volume reduced Event
    """

    obs_id = Field(-1, "observation ID")
    event_id = Field(-1, "event id number")
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
    reference_pulse_shape = Field(
        None, "reference pulse shape for each channel"
    )
    # todo: move to instrument or a static MC container (don't change per
    # event)
    time_slice = Field(0, "width of time slice", unit=u.ns)
    dc_to_pe = Field(None, "DC/PE calibration arrays from MC file")
    pedestal = Field(None, "pedestal calibration arrays from MC file")
    azimuth_raw = Field(
        0, "Raw azimuth angle [radians from N->E] for the telescope"
    )
    altitude_raw = Field(0, "Raw altitude angle [radians] for the telescope")
    azimuth_cor = Field(
        0,
        "the tracking Azimuth corrected for pointing errors for the telescope"
    )
    altitude_cor = Field(
        0,
        "the tracking Altitude corrected for pointing errors for the telescope"
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
    shower_primary_id = Field(None, "MC shower primary ID 0 (gamma), 1(e-),"
                                    "2(mu-), 100*A+Z for nucleons and nuclei,"
                                    "negative for antimatter.")
    tel = Field(
        Map(MCCameraEventContainer), "map of tel_id to MCCameraEventContainer"
    )


class MCHeaderContainer(Container):
    """
    Monte-Carlo information that doesn't change per event
    """
    run_array_direction = Field([], (
        "the tracking/pointing direction in "
        "[radians]. Depending on 'tracking_mode' "
        "this either contains: "
        "[0]=Azimuth, [1]=Altitude in mode 0, "
        "OR "
        "[0]=R.A., [1]=Declination in mode 1."
    ))


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
    az_uncert = Field(0.0, 'reconstructed azimuth uncertainty', unit=u.deg)
    core_x = Field(
        0.0, 'reconstructed x coordinate of the core position', unit=u.m
    )
    core_y = Field(
        0.0, 'reconstructed y coordinate of the core position', unit=u.m
    )
    core_uncert = Field(
        0.0, 'uncertainty of the reconstructed core position', unit=u.m
    )
    h_max = Field(0.0, 'reconstructed height of the shower maximum')
    h_max_uncert = Field(0.0, 'uncertainty of h_max')
    is_valid = Field(False, (
        'direction validity flag. True if the shower direction'
        'was properly reconstructed by the algorithm'
    ))
    tel_ids = Field([], (
        'list of the telescope ids used in the'
        ' reconstruction of the shower'
    ))
    average_size = Field(0.0, 'average size of used')
    goodness_of_fit = Field(0.0, 'measure of algorithm success (if fit)')


class ReconstructedEnergyContainer(Container):
    """
    Standard output of algorithms estimating energy
    """
    energy = Field(-1.0, 'reconstructed energy', unit=u.TeV)
    energy_uncert = Field(-1.0, 'reconstructed energy uncertainty', unit=u.TeV)
    is_valid = Field(False, (
        'energy reconstruction validity flag. True if '
        'the energy was properly reconstructed by the '
        'algorithm'
    ))
    tel_ids = Field([], (
        'array containing the telescope ids used in the'
        ' reconstruction of the shower'
    ))
    goodness_of_fit = Field(0.0, 'goodness of the algorithm fit')


class ParticleClassificationContainer(Container):
    """
    Standard output of gamma/hadron classification algorithms
    """
    # TODO: Do people agree on this? This is very MAGIC-like.
    # TODO: Perhaps an integer classification to support different classes?
    # TODO: include an error on the prediction?
    prediction = Field(0.0, (
        'prediction of the classifier, defined between '
        '[0,1], where values close to 0 are more '
        'gamma-like, and values close to 1 more '
        'hadron-like'
    ))
    is_valid = Field(False, (
        'classificator validity flag. True if the '
        'predition was successful within the algorithm '
        'validity range')
    )

    # TODO: KPK: is this different than the list in the reco
    # container? Why repeat?
    tel_ids = Field([], (
        'array containing the telescope ids used '
        'in the reconstruction of the shower'
    ))
    goodness_of_fit = Field(0.0, 'goodness of the algorithm fit')


class ReconstructedContainer(Container):
    """ collect reconstructed shower info from multiple algorithms """

    shower = Field(
        Map(ReconstructedShowerContainer),
        "Map of algorithm name to shower info"
    )
    energy = Field(
        Map(ReconstructedEnergyContainer),
        "Map of algorithm name to energy info"
    )
    classification = Field(
        Map(ParticleClassificationContainer),
        "Map of algorithm name to classification info"
    )


class TelescopePointingContainer(Container):
    """
    Container holding pointing information for a single telescope
    after all necessary correction and calibration steps.
    These values should be used in the reconstruction to transform
    between camera and sky coordinates.
    """
    azimuth = Field(nan * u.rad, 'Azimuth, measured N->E', unit=u.rad)
    altitude = Field(nan * u.rad, 'Altitude', unit=u.rad)


class DataContainer(Container):
    """ Top-level container for all event information """

    r0 = Field(R0Container(), "Raw Data")
    r1 = Field(R1Container(), "R1 Calibrated Data")
    dl0 = Field(DL0Container(), "DL0 Data Volume Reduced Data")
    dl1 = Field(DL1Container(), "DL1 Calibrated image")
    dl2 = Field(ReconstructedContainer(), "Reconstructed Shower Information")
    mc = Field(MCEventContainer(), "Monte-Carlo data")
    mcheader = Field(MCHeaderContainer(), "Monte-Carlo run header data")
    trig = Field(CentralTriggerContainer(), "central trigger information")
    count = Field(0, "number of events processed")
    inst = Field(InstrumentContainer(), "instrumental information (deprecated")
    pointing = Field(Map(TelescopePointingContainer),
                     'Telescope pointing positions')


class SST1MDataContainer(DataContainer):
    sst1m = Field(SST1MContainer(), "optional SST1M Specific Information")


class NectarCAMCameraContainer(Container):
    """
    Container for Fields that are specific to camera that use zfit
    """
    camera_event_type = Field(int, "camera event type")


    integrals = Field(None, (
        "numpy array containing waveform integrals"
        "(n_channels x n_pixels)"
    ))


    def fill_from_zfile_event(self, event, numTraces):
        self.camera_event_type = event.eventType

        self.integrals = np.array([
            event.hiGain.integrals.gains,
            event.loGain.integrals.gains,
        ])



class NectarCAMContainer(Container):
    """
    Storage for the NectarCAMCameraContainer for each telescope
    """
    tels_with_data = Field([], "list of telescopes with data")
    tel = Field(
        Map(NectarCAMCameraContainer),
        "map of tel_id to NectarCameraContainer")

    def fill_from_zfile_event(self, event, numTraces):
        self.tels_with_data = [event.telescopeID, ]
        nectar_cam_container = self.tel[event.telescopeID]
        nectar_cam_container.fill_from_zfile_event(
            event,
            numTraces,
        )


class NectarCAMDataContainer(DataContainer):
    """
    Data container including NectarCAM information
    """
    nectarcam = Field(NectarCAMContainer(), "NectarCAM Specific Information")

class TargetIOCameraContainer(Container):
    """
    Container for Fields that are specific to cameras that use TARGET
    """
    first_cell_ids = Field(None, ("numpy array of the first_cell_id of each"
                                  "waveform in the camera image (n_pixels)"))


class TargetIOContainer(Container):
    """
    Storage for the TargetIOCameraContainer for each telescope
    """

    tel = Field(Map(TargetIOCameraContainer),
                "map of tel_id to TargetIOCameraContainer")


class TargetIODataContainer(DataContainer):
    """
    Data container including targeto information
    """
    targetio = Field(TargetIOContainer(), "TARGET-specific Data")


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
    tel_id = Field(0, 'telescope identification number')
    ring_center_x = Field(0.0, 'centre (x) of the fitted muon ring')
    ring_center_y = Field(0.0, 'centre (y) of the fitted muon ring')
    ring_radius = Field(0.0, 'radius of the fitted muon ring')
    ring_phi = Field(0.0, 'Orientation of fitted ring')
    ring_inclination = Field(0.0, 'Inclination of fitted ring')
    ring_chi2_fit = Field(0.0, 'chisquare of the muon ring fit')
    ring_cov_matrix = Field(0.0, 'covariance matrix of the muon ring fit')
    ring_containment = Field(0., 'containment of the ring inside the camera')
    ring_fit_method = Field("", 'fitting method used for the muon ring')
    inputfile = Field("", 'input file')


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
    obs_id = Field(0, 'run identification number')
    event_id = Field(0, 'event identification number')
    tel_id = Field(0, 'telescope identification number')
    ring_completeness = Field(0., 'fraction of ring present')
    ring_pix_completeness = Field(0., 'fraction of pixels present in the ring')
    ring_num_pixel = Field(0, 'number of pixels in the ring image')
    ring_size = Field(0., 'size of the ring in pe')
    off_ring_size = Field(0., 'image size outside of ring in pe')
    ring_width = Field(0., 'width of the muon ring in degrees')
    ring_time_width = Field(0., 'duration of the ring image sequence')
    impact_parameter = Field(0.,
                             'distance of muon impact position from centre of mirror')
    impact_parameter_chi2 = Field(0., 'impact parameter chi squared')
    intensity_cov_matrix = Field(0., 'covariance matrix of intensity')
    impact_parameter_pos_x = Field(0., 'impact parameter x position')
    impact_parameter_pos_y = Field(0., 'impact parameter y position')
    COG_x = Field(0.0, 'Centre of Gravity x')
    COG_y = Field(0.0, 'Centre of Gravity y')
    prediction = Field([], 'image prediction')
    mask = Field([], 'image pixel mask')
    optical_efficiency_muon = Field(0., 'optical efficiency muon')
    intensity_fit_method = Field("", 'intensity fit method')
    inputfile = Field("", 'input file')


class HillasParametersContainer(Container):
    intensity = Field(0.0, 'total intensity (size)')

    x = Field(0.0, 'centroid x coordinate')
    y = Field(0.0, 'centroid x coordinate')
    r = Field(0.0, 'radial coordinate of centroid')
    phi = Field(0.0, 'polar coordinate of centroid', unit=u.deg)

    length = Field(0.0, 'RMS spread along the major-axis')
    width = Field(0.0, 'RMS spread along the minor-axis')
    psi = Field(0.0, 'rotation angle of ellipse', unit=u.deg)

    skewness = Field(0.0, 'measure of the asymmetry')
    kurtosis = Field(0.0, 'measure of the tailedness')
