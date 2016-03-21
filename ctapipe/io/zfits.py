# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Components to read ZFITS data.

This requires the protozfitsreader python library to be installed
"""
import logging

import numpy as np
import numpy.ma as ma

from .containers import RawData, RawCameraData
from ctapipe.core import Container

from astropy import units as u
from astropy.coordinates import Angle
from astropy.time import Time

logger = logging.getLogger(__name__)

try:
    import protozfitsreader
except ImportError as err:
    logger.fatal("the `protozfitsreader` python module is required to access MC data: {}"
                 .format(err))
    raise err

__all__ = [
    'zfits_event_source',
]

def zfits_event_source(url, max_events=None, allowed_tels=None):
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

    try:
        zfits = protozfitsreader.ZFile(url)
    except:
        raise RuntimeError("zfits_event_source failed to open '{}'"
                           .format(url))


    counter=0
    eventstream = zfits.move_to_next_event()

    container = Container("zfits_container")
    container.meta.add_item('zfits__input', url)
    container.meta.add_item('zfits__max_events', events)
    container.meta.add_item('pixel_pos', dict())
    container.add_item("dl0", RawData())
    container.add_item("trig", CentralTriggerData())
    container.add_item("count")

    for run_id, event_id in eventstream:
        container.dl0.run_id    = run_id
        container.dl0.event_id  = event_id
        #container.dl0.event_num = zfits.get_event_number()

        # We assume we are in the single-telescope case always:
        container.dl0.tels_with_data     = [zfits.get_telescope_id(), ]
        container.trig.tels_with_trigger = container.dl0.tels_with_data

        time_s, time_ns = zfits.get_central_event_gps_time()
        container.trig.gps_time = Time(time_s * u.s, time_ns * u.ns,
                                       format='gps', scale='utc')

        container.count = counter
        t = np.arange(n_samples)
        
        container.dl0.tel = dict()  # clear the previous telescopes

        # Depecrated loop, we keep it for clarity (similar structure than hessio and mock modules)
        for tel_id in container.dl0.tels_with_data:
            # fill pixel position dictionary, if not already done:
            #TODO: tel_id here is a dummy parameter, we are dealing with single-telescope data!. TBR.
            if tel_id not in container.meta.pixel_pos:
                container.meta.pixel_pos[tel_id] = \
                    zfits.get_pixel_position(tel_id) * u.m

            nchans = zfits.get_num_channels(tel_id)
            container.dl0.tel[tel_id] = RawCameraData(tel_id)
            container.dl0.tel[tel_id].num_channels = nchans
            
            for chan in range(nchans): 
                samples = zfits.get_adc_sample(channel=chan, telescope_id=tel_id)
                integrated = zfits.get_adc_sum(channel=chan, telescope_id=tel_id)
                
                container.dl0.tel[tel_id].pixel_samples[chan] = samples.keys()

                mask = np.zeros(zfits.get_number_of_pixels(),dtype=bool)
                mask[np.array(samples.keys())]=True
                container.dl0.tel[tel_id].adc_samples[chan] = \
                    ma.array(np.array(samples.values()),mask=mask)
                
                mask = np.zeros(zfits.get_number_of_pixels(),dtype=bool)
                mask[np.array(integrated.keys())]=True
                container.dl0.tel[tel_id].adc_integrated[chan] = \
                    ma.array(np.array(integrated.values()),mask=mask)

        yield container
        counter +=1

        if max_events is not None and counter > max_events:
            return

