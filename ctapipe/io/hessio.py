# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Components to read HESSIO data.  

This requires the hessio python library to be installed
"""
import logging

from .containers import RawData, RawCameraData
from ctapipe.core import Container


logger = logging.getLogger(__name__)

try:
    import hessio
except ImportError as err:
    logger.fatal("the `hessio` python module is required to access MC data: {}"
                 .format(err))
    raise err

__all__ = [
    'hessio_event_source',
]


def hessio_event_source(url, max_events=None, single_tel=None):
    """A generator that streams data from an EventIO/HESSIO MC data file
    (e.g. a standard CTA data file.)

    Parameters
    ----------
    url : str
        path to file to open
    max_events : int, optional
        maximum number of events to read
    single_tel : int
        select only a single telescope, if None, all are read. This is
        to emulate the final CTA data format, where there would be 1
        telescope per file (whereas in current monte-carlo, they are
        all interleaved into one file)

    """

    ret = hessio.file_open(url)

    if ret is not 0:
        raise RuntimeError("hessio_event_source failed to open '{}'"
                           .format(url))

    counter = 0
    eventstream = hessio.move_to_next_event()
    container = Container("hessio_container")
    container.meta.add_item('hessio__input', url)
    container.meta.add_item('hessio__max_events', max_events)
    container.meta.add_item('pixel_pos', dict())
    container.add_item("dl0", RawData())
    container.add_item("count")

    for run_id, event_id in eventstream:

        container.dl0.run_id = run_id
        container.dl0.event_id = event_id
        container.dl0.tels_with_data = hessio.get_teldata_list()
        container.count = counter
        
        # handle single-telescope case (ignore others:
        if single_tel is not None:
            if single_tel not in container.dl0.tels_with_data:
                continue
            container.dl0.tels_with_data = [single_tel, ]

        # this should be done in a nicer way to not re-allocate the
        # data each time (right now it's just deleted and garbage
        # collected)

        container.dl0.tel = dict()  # clear the previous telescopes

        for tel_id in container.dl0.tels_with_data:

            # fill pixel position dictionary, if not already done:
            if tel_id not in container.meta.pixel_pos:
                container.meta.pixel_pos[
                    tel_id] = hessio.get_pixel_position(tel_id)

            nchans = hessio.get_num_channel(tel_id)
            container.dl0.tel[tel_id] = RawCameraData(tel_id)
            container.dl0.tel[tel_id].num_channels = nchans

            # load the data per telescope/chan
            for chan in range(nchans):
                container.dl0.tel[tel_id].adc_samples[chan] \
                    = hessio.get_adc_sample(channel=chan,
                                            telescope_id=tel_id)
                container.dl0.tel[tel_id].adc_sums[chan] \
                    = hessio.get_adc_sum(channel=chan,
                                         telescope_id=tel_id)
        yield container
        counter += 1

        if max_events is not None and counter > max_events:
            return
