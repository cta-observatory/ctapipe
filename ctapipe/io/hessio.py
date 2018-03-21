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
def hessio_event_source(url, **kwargs):
    """
    emulate the old `hessio_event_source` generator, using the new
    `HESSIOEventSource` class.  It is preferred to use `HESSIOEventSource` or
    `event_source`, this is only for backward compatibility.


    Parameters
    ----------
    url : str
        path to file to open
    kwargs:
        extra parameters to pass to HESSIOEventSource



    Returns
    -------
    generator:
        a `HESSIOEventSource` wrapped in a generator (for backward
        compatibility)

    """

    reader = HESSIOEventSource(input_url=url, **kwargs)

    return (x for x in reader)

