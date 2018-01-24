# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Components to read HESSIO data.  

This requires the hessio python library to be installed
"""
import logging
import warnings
from astropy import units as u
from astropy.coordinates import Angle
from astropy.time import Time

from .containers import DataContainer
from ..core import Provenance
from ..instrument import TelescopeDescription, SubarrayDescription
from .hessiofilereader import HessioFileReader

logger = logging.getLogger(__name__)


__all__ = [
    'hessio_event_source',
]

def hessio_event_source(url, **params):
    """ emulate the old hessio_event_source generator, using the new
    HessioFileReader.  It is preferred to use HessioFileReader, this is only
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

    reader = HessioFileReader(None, None,
                              input_url=url, **params )


    return (x for x in reader)
