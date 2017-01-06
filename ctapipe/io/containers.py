"""
Container structures for data that should be read or written to disk
"""

from astropy import units as u
from astropy.time import Time

from ..core import Container, Item, Map
from numpy import ndarray

__all__ = ['DataContainer', 'RawDataContainer', 'RawCameraContainer',
           'MCEventContainer', 'MCCameraEventContainer',
           'CalibratedCameraContainer',
           'ReconstructedShowerContainer',
           'ReconstructedEnergyContainer',
           'ParticleClassificationContainer',
           'ReconstructedContainer']

# todo: change some of these Maps to be just 3D NDarrays?


class InstrumentContainer(Container):
    """Storage of header info that does not change with event. This is a
    temporary hack until the Instrument module and database is fully
    implemented.  Eventually static information like this will not be
    part of the data stream, but be loaded and accessed from
    functions.

    """

    telescope_ids = Item([], "list of IDs of telescopes used in the run")
    pixel_pos = Item(Map(ndarray), "map of tel_id to pixel positions")
    optical_foclen = Item(Map(ndarray), "map of tel_id to focal length")
    tel_pos = Item(Map(ndarray), "map of tel_id to telescope position")
    num_pixels = Item(Map(int), "map of tel_id to number of pixels in camera")
    num_samples = Item(Map(int), "map of tel_id to number of time samples")
    num_channels = Item(Map(int), "map of tel_id to number of channels")


class CalibratedCameraContainer(Container):
    """Storage of output of camera calibrationm e.g the final calibrated
    image in intensity units and other per-event calculated
    calibration information.
    """
    calibrated_image = Item(0, "array of camera image", unit=u.electron)
    integration_window = Item(Map(), ("map per channel of bool ndarrays of "
                                      "shape (npix, nsamples) "
                                      "indicating the samples used in "
                                      "the obtaining of the charge, dependant "
                                      "on the integration method used"))
    # todo: rename the following to *_image
    pedestal_subtracted_adc = Item(Map(), ("Map of channel to subtracted "
                                           "ADC image"))
    peakpos = Item(Map(), ("position of the peak as determined by the "
                           "peak-finding algorithm for each pixel"
                           " and channel"))


class CameraCalibrationContainer(Container):
    """
    Storage of externally calculated calibration parameters (not per-event)
    """
    dc_to_pe = Item(None, "DC/PE calibration arrays from MC file")
    pedestal = Item(None, "pedestal calibration arrays from MC file")


class CalibratedContainer(Container):
    """ Calibrated Camera Images and associated data"""
    tel = Item(Map(CalibratedCameraContainer),
               "map of tel_id to CalibratedCameraContainer")


class RawCameraContainer(Container):
    """
    Storage of raw data from a single telescope
    """
    adc_sums = Item(None, ("numpy array containing integrated ADC data "
                           "(n_channels x n_pixels)"))
    adc_samples = Item(None, ("numpy array containing ADC samples"
                              "(n_channels x n_pixels, n_samples)"))


class RawDataContainer(Container):
    """
    Storage of a Merged Raw Data Event
    """

    run_id = Item(-1, "run id number")
    event_id = Item(-1, "event id number")
    tels_with_data = Item([], "list of telescopes with data")
    tel = Item(Map(RawCameraContainer), "map of tel_id to RawCameraContainer")


class MCCameraEventContainer(Container):
    """
    Storage of mc data for a single telescope that change per event
    """
    photo_electron_image = Item(Map(), ("reference image in pure "
                                        "photoelectrons, with no noise"))
    # todo: move to instrument (doesn't change per event)
    reference_pulse_shape = Item(None, ("reference pulse shape for each "
                                        "channel"))
    # todo: move to instrument or a static MC container (don't change per
    # event)
    time_slice = Item(0, "width of time slice", unit=u.ns)
    dc_to_pe = Item(None, "DC/PE calibration arrays from MC file")
    pedestal = Item(None, "pedestal calibration arrays from MC file")
    azimuth_raw = Item(0, "Raw azimuth angle [radians from N->E] "
                          "for the telescope")
    altitude_raw = Item(0, "Raw altitude angle [radians] for the telescope")
    azimuth_cor = Item(0, "the tracking Azimuth corrected for pointing "
                          "errors for the telescope")
    altitude_cor = Item(0, "the tracking Altitude corrected for pointing "
                           "errors for the telescope")


class MCEventContainer(Container):
    """
    Monte-Carlo
    """
    energy = Item(0, "Monte-Carlo Energy")
    alt = Item(0, "Monte-carlo altitude", unit=u.deg)
    az = Item(0, "Monte-Carlo azimuth", unit=u.deg)
    core_x = Item(0, "MC core position")
    core_y = Item(0, "MC core position")
    h_first_int = Item(0, "Height of first interaction")
    tel = Item(Map(MCCameraEventContainer),
               "map of tel_id to MCCameraEventContainer")


class MCHeaderContainer(Container):
    """
    Monte-Carlo information that doesn't change per event
    """
    run_array_direction = Item([], "the tracking/pointing direction in "
                                   "[radians]. Depending on 'tracking_mode' "
                                   "this either contains: "
                                   "[0]=Azimuth, [1]=Altitude in mode 0, "
                                   "OR "
                                   "[0]=R.A., [1]=Declination in mode 1.")


class CentralTriggerContainer(Container):

    gps_time = Item(Time, "central average time stamp")
    tels_with_trigger = Item([], "list of telescopes with data")


class ReconstructedShowerContainer(Container):
    """
    Standard output of algorithms reconstructing shower geometry
    """

    alt = Item(0.0, "reconstructed altitude", unit=u.deg)
    alt_uncert = Item(0.0, "reconstructed altitude uncertainty", unit=u.deg)
    az = Item(0.0, "reconstructed azimuth", unit=u.deg)
    az_uncert = Item(0.0, 'reconstructed azimuth uncertainty', unit=u.deg)
    core_x = Item(0.0, 'reconstructed x coordinate of the core position',
                  unit=u.m)
    core_y = Item(0.0, 'reconstructed y coordinate of the core position',
                  unit=u.m)
    core_uncert = Item(0.0, 'uncertainty of the reconstructed core position',
                       unit=u.m)
    h_max = Item(0.0, 'reconstructed height of the shower maximum')
    h_max_uncert = Item(0.0, 'uncertainty of h_max')
    is_valid = (False, ('direction validity flag. True if the shower direction'
                        'was properly reconstructed by the algorithm'))
    tel_ids = Item([], ('list of the telescope ids used in the'
                        ' reconstruction of the shower'))
    average_size = Item(0.0, 'average size of used')
    goodness_of_fit = Item(0.0, 'measure of algorithm success (if fit)')


class ReconstructedEnergyContainer(Container):
    """
    Standard output of algorithms estimating energy
    """
    energy = Item(-1.0, 'reconstructed energy', unit=u.TeV)
    energy_uncert = Item(-1.0, 'reconstructed energy uncertainty', unit=u.TeV)
    is_valid = Item(False, ('energy reconstruction validity flag. True if '
                            'the energy was properly reconstructed by the '
                            'algorithm'))
    tel_ids = Item([], ('array containing the telescope ids used in the'
                        ' reconstruction of the shower'))
    goodness_of_fit = Item(0.0, 'goodness of the algorithm fit')


class ParticleClassificationContainer(Container):
    """
    Standard output of gamma/hadron classification algorithms
    """
    # TODO: Do people agree on this? This is very MAGIC-like.
    # TODO: Perhaps an integer classification to support different classes?
    # TODO: include an error on the prediction?
    prediction = Item(0.0, ('prediction of the classifier, defined between '
                            '[0,1], where values close to 0 are more '
                            'gamma-like, and values close to 1 more '
                            'hadron-like'))
    is_valid = Item(False, ('classificator validity flag. True if the '
                            'predition was successful within the algorithm '
                            'validity range'))

    # TODO: KPK: is this different than the list in the reco
    # container? Why repeat?
    tel_ids = Item([], ('array containing the telescope ids used '
                        'in the reconstruction of the shower'))
    goodness_of_fit = Item(0.0, 'goodness of the algorithm fit')


class ReconstructedContainer(Container):
    """ collect reconstructed shower info from multiple algorithms """

    shower = Item(Map(ReconstructedShowerContainer),
                  "Map of algorithm name to shower info")
    energy = Item(Map(ReconstructedEnergyContainer),
                  "Map of algorithm name to energy info")
    classification = Item(Map(ParticleClassificationContainer),
                          "Map of algorithm name to classification info")


class DataContainer(Container):
    """ Top-level container for all event information """

    dl0 = Item(RawDataContainer(), "Raw Data")
    dl1 = Item(CalibratedContainer())
    dl2 = Item(ReconstructedContainer(), "Reconstructed Shower Information")
    mc = Item(MCEventContainer(), "Monte-Carlo data")
    mcheader = Item(MCHeaderContainer, "Monte-Carlo run header data")
    trig = Item(CentralTriggerContainer(), "central trigger information")
    count = Item(0, "number of events processed")
    inst = Item(InstrumentContainer(), "instrumental information (deprecated")
