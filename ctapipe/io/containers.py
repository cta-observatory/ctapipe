"""
Container structures for data that should be read or written to disk
"""

from astropy import units as u
from astropy.time import Time

from ..core import Container, Item, Map
from numpy import ndarray

__all__ = ['DataContainer', 'RawDataContainer', 'RawCameraContainer',
           'MCEventContainer', 'MCCameraContainer', 'CalibratedCameraContainer',
           'ReconstructedShowerContainer',
           'ReconstructedEnergyContainer','ParticleClassificationContainer',
           'ReconstructedContainer']

# todo: change some of these Maps to be just 3D NDarrays?


class InstrumentContainer(Container):
    """
    Storage of header info that does not change with event (this is a temporary
     hack until the Instrument module and database is fully implemented.

    This should not be relied upon, as it will be replaced with
    corresponding Instrument module functionality
    """

    pixel_pos = Item(Map(ndarray), "map of tel_id to pixel positions")
    optical_foclen = Item(Map(ndarray), "map of tel_id to focal length")
    tel_pos = Item(Map(ndarray), "map of tel_id to telescope position")
    #TODO: fill these, and remove from cameracontainer metadata
    num_pixels = Item(Map(int), "map of tel_id to number of pixels in camera")
    num_samples = Item(Map(int), "map of tel_id to number of time samples")
    num_channels = Item(Map(int), "map of tel_id to number of channels") 
    
class MCInstrumentContainer(Container):
    """ Storage of Monte-Carlo header information
    pertaining to the instrumental configuration"""
    pass



class CalibratedCameraContainer(Container):
    """Storage of output of camera calibrationm e.g the final calibrated
    image in intensity units and other per-event calculated
    calibration information.
    """
    # todo: rename to pe_image?
    pe_charge = Item(0, "array of camera image in PE")
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
    adc_sums = Item(Map(), ("map of channel to (masked) arrays of all "
                            "integrated ADC data (n_pixels)"))
    adc_samples = Item(Map(), ("map of channel to arrays of "
                               "(n_pixels, n_samples)"))


class RawDataContainer(Container):
    """
    Storage of a Merged Raw Data Event
    """

    run_id = Item(-1, "run id number")
    event_id = Item(-1, "event id number")
    tels_with_data = Item([], "list of telescopes with data")
    tel = Item(Map(RawCameraContainer), "map of tel_id to RawCameraContainer")


# TODO: do these all change per event? If not some should be metadata (headers)
# Rename to MCCameraCalibrationContainer
# Move photo_electrons to MCEventContainer and rename photo_electon_image
class MCCameraContainer(Container):
    """
    Storage of mc data used for a single telescope
    """
    photo_electrons = Item(Map(), ("reference image in pure photoelectrons,"
                                   " with no noise"))
    reference_pulse_shape = Item(Map(), ("map of channel to array "
                                         "defining pulse shape"))
    # TODO: rename to time_slice_width? and move to RawCameraContainer
    time_slice = Item(0, "width of time slice")
    dc_to_pe = Item(None, "DC/PE calibration arrays from MC file")
    pedestal = Item(None, "pedestal calibration arrays from MC file")


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
    tel = Item(Map(MCCameraContainer), "map of tel_id to MCCameraContainer")


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
    # TODO: Perhaps an integer classification + error?
    prediction = Item(0.0, ('prediction of the classifier, defined between '
                            '[0,1], where values close to 0 are more gamma-like,'
                            ' and values close to 1 more hadron-like'))
    is_valid = Item(False, ('classificator validity flag. True if the predition '
                            'was successful within the algorithm validity range'))

    # TODO: KPK: is this different than the list in the reco container? Why repeat?
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



# dl0.event.tel[5]
# dl0.event.sub.trig
# dl0.mon.tel[5]

# class RawEventContainer(Container):
#     tel = Item(Map(RawEventTelescopeContainer(), "map by tel_id")
#     sub = Item(RawEventSubarrayContainer(),"per-subarray event info"))
#
# class RawMonitorContainer(Container):
#
#
# class RawDataContainer(Container):
#     """ Data Level 0 Container """
#
#     event = Item(RawEventContainer(), "event-wise data")
#     mon = Item
#
#     pass
#
# class ProcessedDataContainer(Container):
#     """ Data Level 1 Container"""
#
#     pass
#
# class ReconstructedDataContainer(Container):
#     """ Data Level 2 Container """
#     pass
#
# class ReducedDataContainer(Container):
#     """ Data Level 3 Container """
#     pass
#
#
# class MainDataContainer(Container):
#     dl0 = Item(RawDataContainer(), "Raw (DL0) Data")
#     dl1 = Item(ProcessedDataContainer(), "Processed (DL1) data")
#     dl2 = Item(ReconstructedDataContainer(), "Reconstructed (DL2) data")
#     dl3 = Item(ReducedDataContainer(), "Reduced (DL3) science data")

class DataContainer(Container):
    """ Top-level container for all event information """

    dl0 = Item(RawDataContainer(), "Raw Data")
    dl1 = Item(CalibratedContainer())
    dl2 = Item(ReconstructedContainer(), "Reconstructed Shower Information")
    mc = Item(MCEventContainer(), "Monte-Carlo data")
    trig = Item(CentralTriggerContainer(), "central trigger information")
    count = Item(0, "number of events processed")
    inst = Item(InstrumentContainer(), "instrumental information (deprecated")
