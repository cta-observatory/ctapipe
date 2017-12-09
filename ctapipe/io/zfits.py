# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Components to read ZFITS data.

This requires the protozfitsreader python library to be installed
"""
import logging
from .containers import DataContainer
from .containers import (
    DigiCamCameraContainer,
    DigiCamExpertCameraContainer,
)

logger = logging.getLogger(__name__)

try:
    import protozfitsreader
except ImportError as err:
    logger.fatal(
        "the `protozfitsreader` python module is required: {}"
        .format(err)
    )
    raise err

__all__ = [
    'zfits_event_source',
]


def zfits_event_source(
        url,
        max_events=None,
        allowed_tels=None,
        expert_mode=False):
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
        file = protozfitsreader.ZFile(url)
    except:
        raise RuntimeError("zfits_event_source failed to open '{}'"
                           .format(url))

    eventstream = file.move_to_next_event()
    for counter, (run_id, event_id) in enumerate(eventstream):
        # define the main container and fill some metadata
        data = DataContainer()
        data.meta['zfits__input'] = url
        data.meta['zfits__max_events'] = max_events
        data.r0.run_id = run_id
        data.r0.event_id = event_id
        data.count = counter

        data.r0.tels_with_data = remove_forbidden_telescopes(
            [file.event.telescopeID, ],
            allowed_tels
        )

        assert number_of_pixels(file) == 1296

        for tel_id in data.r0.tels_with_data:
            data.inst.num_channels[tel_id] = file.event.num_gains
            data.inst.num_pixels[tel_id] = number_of_pixels(file)

            container = ContainerFactory(expert_mode)()
            container = fill_container_somehow(
                container, file, tel_id)
            if expert_mode:
                container = fill_container_if_expert_mode(
                    container, file, tel_id)

            data.r0.tel[tel_id] = container
        yield data

    if max_events is not None and counter > max_events:
        return


def number_of_pixels(file):
    return len(
        file._get_numpyfield(
            file.event.hiGain.waveforms.pixelsIndices
        )
    )


def remove_forbidden_telescopes(tels_with_data, allowed_tels):
    if allowed_tels:
        return [
            [
                tel_id for tel_id in tels_with_data
                if tel_id in sublist
            ]
            for sublist in allowed_tels
        ]
    else:
        return tels_with_data


def ContainerFactory(expert_mode):
    if not expert_mode:
        return DigiCamCameraContainer
    else:
        return DigiCamExpertCameraContainer


def fill_container_somehow(container, file, tel_id):

    container.camera_event_number = file.event.eventNumber
    container.pixel_flags = file.get_pixel_flags(telescope_id=tel_id)

    seconds, nano_seconds = file.get_local_time()
    container.local_camera_clock = seconds * 1e9 + nano_seconds

    container.event_type = file.get_event_type()
    container.eventType = file.get_eventType()

    container.num_samples = (
        file._get_numpyfield(file.event.hiGain.waveforms.samples).shape[0] //
        file._get_numpyfield(file.event.hiGain.waveforms.pixelsIndices).shape[0]
    )
    container.adc_samples = file.get_adcs_samples(telescope_id=tel_id)
    return container


def fill_container_if_expert_mode(container, file, tel_id):
    ''' if expert_mode: '''
    container.trigger_input_traces = file.get_trigger_input_traces(telescope_id=tel_id)
    container.trigger_output_patch7 = file.get_trigger_output_patch7(telescope_id=tel_id)
    container.trigger_output_patch19 = file.get_trigger_output_patch19(telescope_id=tel_id)
    return container
