"""
Module containing general functions that will calibrate any event regardless of
the source/telescope, and store the calibration inside the event container.
"""


from copy import copy
from .mc import calibrate_mc, set_integration_correction
from functools import partial
import logging
from ctapipe.io.containers import RawData, CalibratedCameraData
from ctapipe.io import CameraGeometry

logger = logging.getLogger(__name__)


def calibrate_event(event, params, geom_dict=None):
    """
    Generic calibrator for events. Calls the calibrator corresponding to the
    source of the event, and stores the dl1 (pe_charge) information into a
    new event container.

    Parameters
    ----------
    event : container
        A `ctapipe` event container
    params : dict
        REQUIRED:

        params['integrator'] - Integration scheme

        params['window'] - Integration window size

        params['shift'] - Starting sample for this integration

        (adapted such that window fits into readout).

        OPTIONAL:

        params['clip_amp'] - Amplitude in p.e. above which the signal is
        clipped.

        params['calib_scale'] : Identical to global variable CALIB_SCALE in
        reconstruct.c in hessioxxx software package. 0.92 is the default value
        (corresponds to HESS). The required value changes between cameras
        (GCT = 1.05).

        params['sigamp'] - Amplitude in ADC counts above pedestal at which a
        signal is considered as significant (separate for high gain/low gain).
    geom_dict : dict[`ctapipe.io.CameraGeometry`]
        Dict of pixel geometry for each telescope. Leave as None for automatic
        calculation when it is required.

    Returns
    -------
    calibrated : container
        A new `ctapipe` event container containing the dl1 information, and a
        reference to all other information contained in the original event
        container.
    """

    # Obtain relevent calibrator
    switch = {
        'hessio':
            partial(calibrate_mc, event=event, params=params)
        }
    try:
        calibrator = switch[event.meta.source]
    except KeyError as e:
        logger.exception("unknown event source '{}'".format(event.meta.source))
        raise

    calibrated = copy(event)

    # Add dl1 to the event container (if it hasn't already been added)
    try:
        calibrated.add_item("dl1", RawData)
        calibrated.dl1.run_id = event.dl0.run_id
        calibrated.dl1.event_id = event.dl0.event_id
        calibrated.dl1.tels_with_data = event.dl0.tels_with_data
    except AttributeError:
        pass

    # Fill dl1
    calibrated.dl1.tel = dict()  # clear the previous telescopes
    for telid in event.dl0.tels_with_data:
        nchan = event.dl0.tel[telid].num_channels
        npix = event.dl0.tel[telid].num_pixels
        calibrated.dl1.tel[telid] = CalibratedCameraData(telid)
        calibrated.dl1.tel[telid].num_channels = nchan
        calibrated.dl1.tel[telid].num_pixels = npix

        geom = geom_dict[telid] if geom_dict is not None else None

        pe, window = calibrator(telid=telid, geom=geom)
        for chan in range(nchan):
            calibrated.dl1.tel[telid].pe_charge[chan] = pe[chan]
            calibrated.dl1.tel[telid].integration_window[chan] = window[chan]

    return calibrated


def calibrate_source(source, params):
    """
    Generator for calibrating all events in a file. Using this function is
    faster than `calibrate_event` if you require more than one event
    calibrated, as the geometry for each telescope is stored instead of being
    recalculated. If you only require one event calibrated, use
    `calibrate_event`.

    Parameters
    ----------
    source : generator
        A `ctapipe` event generator such as
        `ctapipe.io.hessio.hessio_event_source`
    params : dict
        REQUIRED:

        params['integrator'] - Integration scheme

        params['window'] - Integration window size

        params['shift'] - Starting sample for this integration

        (adapted such that window fits into readout).

        OPTIONAL:

        params['clip_amp'] - Amplitude in p.e. above which the signal is
        clipped.

        params['calib_scale'] : Identical to global variable CALIB_SCALE in
        reconstruct.c in hessioxxx software package. 0.92 is the default value
        (corresponds to HESS). The required value changes between cameras
        (GCT = 1.05).

        params['sigamp'] - Amplitude in ADC counts above pedestal at which a
        signal is considered as significant (separate for high gain/low gain).

    Returns
    -------
    calibrated : container
        A new `ctapipe` event container containing the dl1 information, and a
        reference to all other information contained in the original event
        container.
    """
    geom_dict = {}

    for event in source:
        # Fill dict so geom are only calculated once per telescope
        # TODO: create check if geom is even needed for integrator
        for telid in event.dl0.tels_with_data:
            if telid not in geom_dict:
                geom = CameraGeometry.guess(*event.meta.pixel_pos[telid],
                                            event.meta.optical_foclen[telid])
                geom_dict[telid] = geom

        calibrated = calibrate_event(event, params, geom_dict)

        yield calibrated

