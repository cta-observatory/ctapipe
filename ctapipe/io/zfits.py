# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Components to read ZFITS data.

This requires the protozfitsreader python library to be installed
"""
import logging
from .containers import DataContainer

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
    """A generator that streams data from an ZFITs data file


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

    # load the zfits file
    try:
        zfits = protozfitsreader.ZFile(url)
    except:
        raise RuntimeError("zfits_event_source failed to open '{}'"
                           .format(url))

    # intialise counter and event generator
    counter = 0
    eventstream = zfits.move_to_next_event()

    # loop over the events
    for run_id, event_id in eventstream:
        # define the main container and fill some metadata
        data = DataContainer()
        data.meta['zfits__input'] = url
        data.meta['zfits__max_events'] = max_events
        data.dl0.run_id = run_id
        data.dl0.event_id = event_id
        data.dl0.tels_with_data = [zfits.event.telescopeID, ]
        data.count = counter
        # remove forbidden telescopes
        if allowed_tels:
            data.dl0.tels_with_data = \
                [list(filter(lambda x: x in data.dl0.tels_with_data, sublist)) for sublist in allowed_tels]

        for tel_id in data.dl0.tels_with_data :
            # TODO: add the time flag
            data.dl0.tel[tel_id].event_number = zfits.event.eventNumber
            data.dl0.tel[tel_id].num_channels =  zfits.event.num_gains
            data.dl0.tel[tel_id].num_pixels = zfits._get_numpyfield(zfits.event.hiGain.waveforms.pixelsIndices).shape[0]
            data.dl0.tel[tel_id].num_samples = zfits._get_numpyfield(zfits.event.hiGain.waveforms.samples).shape[0] //\
                                               zfits._get_numpyfield(zfits.event.hiGain.waveforms.pixelsIndices).shape[0]
            data.dl0.tel[tel_id].adc_samples = zfits.get_adcs_samples(telescope_id=tel_id)
            data.dl0.tel[tel_id].pixel_flags = zfits.get_pixel_flags(telescope_id=tel_id)
            #

        yield data
        counter += 1

    if max_events is not None and counter > max_events:
        return
