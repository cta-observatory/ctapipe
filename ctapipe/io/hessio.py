# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Backward compatibility function for reading hessio files.

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
    """
    emulate the old `hessio_event_source` generator, using the new
    `HESSIOEventSource` class.  It is preferred to use `HESSIOEventSource` or
    `event_source`, this is only for backward compatibility.


    Parameters
    ----------
    url : str
        path to file to open
    max_events : int, optional
        maximum number of events to read
    allowed_tels : List[int]
        select only a subset of telescope, if None, all are read. This can be
        used for example emulate the final CTA data format, where there would
        be 1 telescope per file (whereas in current monte-carlo, they are all
        interleaved into one file)


    Returns
    -------
    generator:
        a `HESSIOEventSource` wrapped in a generator (for backward
        compatibility)

    """

    reader = HESSIOEventSource(None, None,
                               input_url=url, **params)


    return (x for x in reader)

