# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Components to read HESSIO data.  

This requires the hessio python library to be installed
"""
import logging

from .containers import RawData
from .containers import RawCameraData, MCShowerData, CentralTriggerData
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

    counter = 0
    eventstream = pyhessio.move_to_next_event()
    if allowed_tels is not None:
        allowed_tels = set(allowed_tels)
    container = Container("hessio_container")
    container.meta.add_item('hessio__input', url)
    container.meta.add_item('hessio__max_events', max_events)
    container.meta.add_item('pixel_pos', dict())
    container.meta.add_item('optical_foclen', dict())
    container.add_item("dl0", RawData())
    container.add_item("mc", MCShowerData())
    container.add_item("trig", CentralTriggerData())
    container.add_item("count")

    for run_id, event_id in eventstream:

        container.dl0.run_id = run_id
        container.dl0.event_id = event_id
        container.dl0.tels_with_data = set(pyhessio.get_teldata_list())

        # handle telescope filtering by taking the intersection of
        # tels_with_data and allowed_tels
        if allowed_tels is not None:
            selected = container.dl0.tels_with_data & allowed_tels
            if len(selected) == 0:
                continue  # skip event
            container.dl0.tels_with_data = selected

        container.trig.tels_with_trigger \
            = pyhessio.get_central_event_teltrg_list()
        time_s, time_ns = pyhessio.get_central_event_gps_time()
        container.trig.gps_time = Time(time_s * u.s, time_ns * u.ns,
                                       format='gps', scale='utc')
        container.mc.energy = pyhessio.get_mc_shower_energy() * u.TeV
        container.mc.alt = Angle(pyhessio.get_mc_shower_altitude(), u.rad)
        container.mc.az = Angle(pyhessio.get_mc_shower_azimuth(), u.rad)
        container.mc.core_x = pyhessio.get_mc_event_xcore() * u.m
        container.mc.core_y = pyhessio.get_mc_event_ycore() * u.m

        container.count = counter

        # this should be done in a nicer way to not re-allocate the
        # data each time (right now it's just deleted and garbage
        # collected)

        container.dl0.tel = dict()  # clear the previous telescopes

        for tel_id in container.dl0.tels_with_data:

            # fill pixel position dictionary, if not already done:
            if tel_id not in container.meta.pixel_pos:
                container.meta.pixel_pos[tel_id] \
                    = pyhessio.get_pixel_position(tel_id) * u.m
                container.meta.optical_foclen[tel_id] = pyhessio.get_optical_foclen(tel_id) * u.m;

            nchans = pyhessio.get_num_channel(tel_id)
            container.dl0.tel[tel_id] = RawCameraData(tel_id)
            container.dl0.tel[tel_id].num_channels = nchans

            # load the data per telescope/chan
            for chan in range(nchans):
                container.dl0.tel[tel_id].adc_samples[chan] \
                    = pyhessio.get_adc_sample(channel=chan,
                                              telescope_id=tel_id)
                container.dl0.tel[tel_id].adc_sums[chan] \
                    = pyhessio.get_adc_sum(channel=chan,
                                           telescope_id=tel_id)
        yield container
        counter += 1

        if max_events is not None and counter > max_events:
            return
