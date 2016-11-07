"""
Container structures for data that should be read or written to disk
"""

from astropy import units as u
from astropy.time import Time

from ..core import Container, Item, Map
from numpy import ndarray

__all__ = ['EventContainer', 'RawDataContainer', 'RawCameraContainer',
           'MCEventContainer', 'MCCameraContainer',
           'CalibratedCameraContainer', 'ReconstructedShowerContainer']

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


class CalibratedCameraContainer(Container):
    """
    Storage of calibrated (p.e.) data from a single telescope
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

class CalibratedContainer(Container):
    """ Calibrated Camera Images and associated data"""
    tel = Item(Map(CalibratedCameraContainer), "map of tel_id to CalibratedCameraContainer")


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
class MCCameraContainer(Container):
    """
    Storage of mc data used for a single telescope
    """
    photo_electrons = Item(Map(), "map of channel to PE")
    refshapes = Item(Map(), "map of channel to array defining pulse shape")
    refstep = Item(0, "RENAME AND WRITE DESC FOR THIS!")
    lrefshape = Item(0, "RENAME AND WRITE DESC FOR THIS!")
    # TODO: rename to time_slice_width?
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

    
    alt  = Item(0.0,"reconstructed altitude", unit=u.deg)
    alt_uncert = Item(0.0,"reconstructed altitude uncertainty", unit=u.deg)
    az = Item(0.0,"reconstructed azimuth", unit=u.deg)
    az_uncert =Item(0.0,'reconstructed azimuth uncertainty', unit=u.deg)
    core_x = Item(0.0,'reconstructed x coordinate of the core position', unit=u.m)
    core_y = Item(0.0,'reconstructed y coordinate of the core position',unit=u.m)
    core_uncert = Item(0.0,'uncertainty of the reconstructed core position',unit=u.m)
    h_max = Item(0.0,'reconstructed height of the shower maximum')
    h_max_uncert=Item(0.0,'uncertainty of h_max')
    is_valid=(False,('direction validity flag. True if the shower direction'
                     'was properly reconstructed by the algorithm'))
    tel_ids = Item([],('list of the telescope ids used in the'
                       ' reconstruction of the shower'))
    average_size = Item(0.0, 'average size of used')

    
class ReconstructedEnergyContainer(Container):
    """
    Standard output of algorithms estimating energy
    """
    energy = Item(-1.0, 'reconstructed energy', unit=u.TeV)
    energy_uncert= Item(-1.0, 'reconstructed energy uncertainty', unit=u.TeV)
    is_valid = Item(False,('energy reconstruction validity flag. True if '
                           'the energy was properly reconstructed by the algorithm'))
    tel_ids = Item([], ('array containing the telescope ids used in the'
                        ' reconstruction of the shower'))
    goodness_of_fit = Item(0.0, 'goodness of the algorithm fit')

    
class ParticleClassificationContainer(Container):
    """
    Standard output of gamma/hadron classification algorithms
    """
    #TODO: Do people agree on this? This is very MAGIC-like.
    #TODO: Perhaps an integer classification + error?
    prediction = Item(0.0, ('prediction of the classifier, defined between '
                            '[0,1], where values close to 0 are more gamma-like,'
                            ' and values close to 1 more hadron-like')) 
    is_valid = Item(False, ('classificator validity flag. True if the predition '
                            'was successful within the algorithm validity range'))
                    
    #TODO: KPK: is this different than the list in the reco container? Why repeat?
    tel_ids  = Item([],('array containing the telescope ids used '
                        'in the reconstruction of the shower'))
    goodness_of_fit  = Item(0.0,'goodness of the algorithm fit')

            
class EventContainer(Container):
    """ Top-level container for all event information """

    dl0 = Item(RawDataContainer(), "Raw Data")
    dl1 = Item(CalibratedContainer())
    mc = Item(MCEventContainer(), "Monte-Carlo data")
    trig = Item(CentralTriggerContainer(), "central trigger information")
    count = Item(0, "number of events processed")
    inst = Item(InstrumentContainer(), "instrumental information (deprecated")
