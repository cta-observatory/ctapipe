# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Components to read HESSIO data.  

This requires the hessio python library to be installed
"""
import logging

from .hessioeventsource import HESSIOEventSource
from astropy.utils.decorators import deprecated

logger = logging.getLogger(__name__)


__all__ = [
    'hessio_event_source',
]

@deprecated(0.5, message="prefer the use of an EventSource or "
                         "EventSourceFactory")
def hessio_event_source(url, **params):
    """ emulate the old hessio_event_source generator, using the new
    HESSIOEventSource.  It is preferred to use HESSIOEventSource, this is only
    for backward compatibility.

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
    requested_event : int
        Seek to a paricular event index
    use_event_id : bool
        If True ,'requested_event' now seeks for a particular event id instead
        of index


    """

    reader = HESSIOEventSource(None, None,
                               input_url=url, **params)


    return (x for x in reader)
