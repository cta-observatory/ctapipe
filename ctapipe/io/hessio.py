"""
Components to read HESSIO data.  

This requires the hessio python library to be installed
"""

from ctapipe.core import  Container
from collections import defaultdict

import logging
logger = logging.getLogger(__name__)

try:
    import hessio
except ImportError as err:
    logger.fatal("the `hessio` python module is required to access MC data: {}"
                 .format(err))
    raise err


def hessio_event_source(url, max_events=None):
    """A generator that streams data from an EventIO/HESSIO MC data file
    (e.g. a standard CTA data file.)

    Parameters
    ----------
    url: string
        path to file to open
    max_events: int, optional
        maximum number of events to read
    """

    ret = hessio.file_open(url)

    if ret is not 0:
        raise RuntimeError("hessio_event_source failed to open '{}'"
                           .format(url))

    counter = 0
    eventstream = hessio.move_to_next_event()
    data = Container("hessio_data")
    data.add_item("run_id")
    data.add_item("event_id")
    data.add_item("tels_with_data")
    data.add_item("data")
    data.add_item("num_channels")
    
    for run_id, event_id in eventstream:

        data.run_id = run_id
        data.event_id = event_id
        data.tels_with_data = hessio.get_teldata_list()

        # this should be done in a nicer way to not re-allocate
        # the data each time

        data.data = defaultdict(dict)
        data.num_channels = defaultdict(int)

        for tel_id in data.tels_with_data:
            data.num_channels = hessio.get_num_channel(tel_id)
            for chan in range(data.num_channels+1):
                data.data[tel_id][chan] \
                    = hessio.get_pixel_data(channel=chan,
                                            telescopeId=tel_id)
        yield data
        counter += 1

        if counter > max_events:
            return
    
