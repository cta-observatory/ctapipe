"""
"""

from astropy import units as u
from astropy.time import Time

from ..core import Container, Item, Map

__all__ = ['EventContainer','RawDataContainer', 'RawCameraContainer',
           'MCShowerContainer', 'MCEventContainer', 'MCCameraContainer',
           'CalibratedCameraContainer']

# todo: change some of these Maps to be just 3D NDarrays?
class CalibratedCameraContainer(Container):
    """
    Storage of calibrated (p.e.) data from a single telescope
    """
    pe_charge = Item(0, "array of camera image in PE") # todo: rename to pe_image?
    integration_window = Item(Map(), ("map per channel of bool ndarrays of "
                                      "shape (npix, nsamples) "
                                      "indicating the samples used in "
                                      "the obtaining of the charge, dependant on the "
                                      "integration method used"))
    # todo: rename the following to *_image
    pedestal_subtracted_adc = Item(Map(), "Map of channel  to subtracted ADC image")
    peakpos = Item(Map(), ("position of the peak as determined by the peak-finding"
                           " algorithm for each pixel and channel"))

    num_channels = Item(0, "number of channels")  # todo: this is metadata, not a column?
    num_pixels = Item(0, "number of channels")  # todo: this is metadata, not a column?

    # todo: this cannot be written to a table, so needs to be metadata. Do they change per event?
    calibration_parameters = Item(dict(), "parameters used to calbrate the event")


class RawCameraContainer(Container):
    """
    Storage of raw data from a single telescope
    """
    adc_sums = Item(Map(), ("map of channel to (masked) arrays of all "
                            "integrated ADC data (n_pixels)"))
    adc_samples = Item(Map(), "map of channel to arrays of (n_pixels, n_samples)")
    pedestal = Item(0, "Pedestal values")
    num_channels = Item(0,"Number of channels in camera") # TODO: this is metadata
    num_pixels = Item(0,"Number of pixels in camera") # TODO: this is metadata and not needed
    num_samples = Item(0,"Number of samples per channel)") #TODO: this is metadata

class RawDataContainer(Container):
    """
    Storage of a Merged Raw Data Event
    """

    run_id = Item(-1, "run id number")
    event_id = Item(-1, "event id number")
    tels_with_data = Item([], "list of telescopes with data")
    tel = Item(Map(), "map of tel_id to RawCameraContainer")
        

# TODO: this should be replaced with a standard Shower container (and just have one called "mc" in the event)
class MCShowerContainer(Container):
    """
    Monte-Carlo shower representation
    """
    energy = Item(0, "Monte-Carlo Energy")
    alt = Item(0, "Monte-carlo altitude", unit=u.deg)
    az = Item(0, "Monte-Carlo azimuth", unit=u.deg)
    core_x = Item(0, "MC core position")
    core_y = Item(0, "MC core position")
    h_first_int = Item(0, "Height of first interaction")

# TODO: why is this a subclass of MCShowerContainer?
class MCEventContainer(MCShowerContainer):
    """
    Monte-Carlo
    """
    tel = Item(Map(), "map of tel_id to MCCameraContainer")


class CentralTriggerContainer(Container):

    gps_time = Item(Time, "central average time stamp")
    tels_with_trigger = Item([], "list of telescopes with data")


#TODO: do these all change per event? If not some should be metadata (headers)
class MCCameraContainer(Container):
    """
    Storage of mc data used for a single telescope
    """
    photo_electrons = Item(Map(), "map of channel to PE")
    refshapes = Item(Map(), "map of channel to array defining pulse shape")
    refstep = Item(0, "RENAME AND WRITE DESC FOR THIS!")
    lrefshape = Item(0, "RENAME AND WRITE DESC FOR THIS!")
    time_slice = Item(0, "width of time slice") # TODO: rename to time_slice_width?





class EventContainer(Container):
    """ Top-level container for all event information """

    dl0 = Item(RawCameraContainer(), "Raw Data")
    dl1 = Item(CalibratedCameraContainer())
    mc = Item(MCEventContainer(), "Monte-Carlo data")
    trig = Item(CentralTriggerContainer(), "central trigger information")
    count = Item(0, "number of events processed")

    #self.meta.add_item('tel_pos', dict())
    #self.meta.add_item('pixel_pos', dict())
    #self.meta.add_item('optical_foclen', dict())
    #self.meta.add_item('source', "unknown")


