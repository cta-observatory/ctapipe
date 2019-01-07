from ctapipe.core.factory import Factory
from ctapipe.io.eventsource import EventSource

# EventFileReader imports so that EventFileReaderFactory can see them
# (they need to exist in the global namespace)
import ctapipe.io.hessioeventsource
from . import sst1meventsource
from . import nectarcameventsource
from . import lsteventsource
import ctapipe.io.targetioeventsource


__all__ = ['event_source']


def event_source(input_url, config=None, parent=None, **kwargs):
    """
    Helper function to quickly construct an `EventSourceFactory` and produce
    an `EventSource`. This may be used in small scripts and demos for
    simplicity. In a `ctapipe.core.Tool` class, a `EventSourceFactory` should
    be manually constructed, so that the configuration info is correctly
    passed in.

    Examples
    --------
    >>> with event_source(url) as source:
    >>>    for event in source:
    >>>         print(event.r0.event_id)

    Parameters
    ----------
    input_url: str
        filename or URL pointing to an event file.

    Returns
    -------
    EventSource:
        a properly constructed `EventSource` subclass, depending on the
        input filename.
    """

    reader = EventSource.from_url(
        input_url,
        config,
        parent,
        **kwargs)

    return reader
