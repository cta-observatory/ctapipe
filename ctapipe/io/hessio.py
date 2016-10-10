# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Components to read HESSIO data.  

This requires the hessio python library to be installed
"""
import logging

from .containers import EventContainer, RawData
from .containers import RawCameraData, MCEvent, MCCamera, CentralTriggerData
from ctapipe.core import Container

from astropy import units as u
from astropy.coordinates import Angle
from astropy.time import Time

logger = logging.getLogger(__name__)

try:
    import pyhessio
except ImportError as err:
    logger.fatal("the `pyhessio` python module is required to access MC data: {}"
                 .format(err))
    raise err

__all__ = [
    'hessio_event_source',
]


def hessio_event_source(url, max_events=None, allowed_tels=None):
    """A generator that streams data from an EventIO/HESSIO MC data file
    (e.g. a standard CTA data file.)

    Parameters
    ----------
    url : str
        path to file to open
    max_events : int, optional
        maximum number of events to read
    allowed_tels : list[int]
        select only a subset of telescope, if None, all are read. This can
        be used for example emulate the final CTA data format, where there
        would be 1 telescope per file (whereas in current monte-carlo,
        they are all interleaved into one file)

    """

    ret = pyhessio.file_open(url)

    if ret is not 0:
        raise RuntimeError("hessio_event_source failed to open '{}'"
                           .format(url))

    # the container is initialized once, and data is replaced within
    # it after each yield

    counter = 0
    eventstream = pyhessio.move_to_next_event()
    if allowed_tels is not None:
        allowed_tels = set(allowed_tels)
    event = EventContainer()
    event.meta.source = "hessio"

    # some hessio_event_source specific parameters
    event.meta.add_item('hessio__input', url)
    event.meta.add_item('hessio__max_events', max_events)

    for run_id, event_id in eventstream:

        event.dl0.run_id = run_id
        event.dl0.event_id = event_id
        event.dl0.tels_with_data = set(pyhessio.get_teldata_list())
        
        # handle telescope filtering by taking the intersection of
        # tels_with_data and allowed_tels
        if allowed_tels is not None:
            selected = event.dl0.tels_with_data & allowed_tels
            if len(selected) == 0:
                continue  # skip event
            event.dl0.tels_with_data = selected

        event.trig.tels_with_trigger \
            = pyhessio.get_central_event_teltrg_list()
        time_s, time_ns = pyhessio.get_central_event_gps_time()
        event.trig.gps_time = Time(time_s * u.s, time_ns * u.ns,
                                   format='gps', scale='utc')
        event.mc.energy = pyhessio.get_mc_shower_energy() * u.TeV
        event.mc.alt = Angle(pyhessio.get_mc_shower_altitude(), u.rad)
        event.mc.az = Angle(pyhessio.get_mc_shower_azimuth(), u.rad)
        event.mc.core_x = pyhessio.get_mc_event_xcore() * u.m
        event.mc.core_y = pyhessio.get_mc_event_ycore() * u.m
        event.mc.h_first_int = pyhessio.get_mc_shower_h_first_int() * u.m

        event.count = counter

        # this should be done in a nicer way to not re-allocate the
        # data each time (right now it's just deleted and garbage
        # collected)

        event.dl0.tel = dict()  # clear the previous telescopes
        event.mc.tel = dict()  # clear the previous telescopes

        for tel_id in event.dl0.tels_with_data:

            # fill pixel position dictionary, if not already done:
            if tel_id not in event.meta.pixel_pos:
                event.meta.pixel_pos[tel_id] \
                    = pyhessio.get_pixel_position(tel_id) * u.m
                event.meta.optical_foclen[
                    tel_id] = pyhessio.get_optical_foclen(tel_id) * u.m

            # fill telescope position dictionary, if not already done:
            if tel_id not in event.meta.tel_pos:
                event.meta.tel_pos[
                    tel_id] = pyhessio.get_telescope_position(tel_id) * u.m

            nchans = pyhessio.get_num_channel(tel_id)
            npix = pyhessio.get_num_pixels(tel_id)
            nsamples = pyhessio.get_num_samples(tel_id)
            event.dl0.tel[tel_id] = RawCameraData(tel_id)
            event.dl0.tel[tel_id].num_channels = nchans
            event.dl0.tel[tel_id].num_pixels = npix
            event.dl0.tel[tel_id].num_samples = nsamples
            event.mc.tel[tel_id] = MCCamera(tel_id)

            event.dl0.tel[tel_id].calibration \
                = pyhessio.get_calibration(tel_id)
            event.dl0.tel[tel_id].pedestal \
                = pyhessio.get_pedestal(tel_id)

            # load the data per telescope/chan
            for chan in range(nchans):
                event.dl0.tel[tel_id].adc_samples[chan] \
                    = pyhessio.get_adc_sample(channel=chan,
                                              telescope_id=tel_id)
                event.dl0.tel[tel_id].adc_sums[chan] \
                    = pyhessio.get_adc_sum(channel=chan,
                                           telescope_id=tel_id)
                event.mc.tel[tel_id].refshapes[chan] = \
                    pyhessio.get_ref_shapes(tel_id, chan)

            # load the data per telescope/pixel
            event.mc.tel[tel_id].photo_electrons \
                = pyhessio.get_mc_number_photon_electron(telescope_id=tel_id)
            event.mc.tel[tel_id].refstep = pyhessio.get_ref_step(tel_id)
            event.mc.tel[tel_id].lrefshape = pyhessio.get_lrefshape(tel_id)
            event.mc.tel[tel_id].time_slice = \
                pyhessio.get_time_slice(tel_id)
        yield event
        counter += 1

        if max_events is not None and counter >= max_events:
            return
