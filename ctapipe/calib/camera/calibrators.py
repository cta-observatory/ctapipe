"""
Module containing general functions that will calibrate any event regardless of
the source/telescope, and store the calibration inside the event container.
"""
import argparse
from copy import copy
from ctapipe.calib.camera import mc
from ctapipe.calib.camera.integrators import integrator_dict, \
    integrators_requiring_geom
from functools import partial
from ctapipe.io.containers import RawDataContainer, CalibratedCameraContainer
from ctapipe.io import CameraGeometry
from astropy import log


def calibration_parser(origin):
    """
    Obtain the correct parser for your input file.

    Parameters
    ----------
    origin : str
        Origin of data file e.g. hessio

    Returns
    -------
    parser : `astropy.utils.compat.argparse.ArgumentParser`
        Argparser for calibration arguments.
    ns : `argparse.Namespace`
        Namespace containing the correction for default values so they use
        a custom Action.
    """

    # Obtain relevent calibrator arguments
    switch = {
        'hessio':
            lambda: mc.calibration_arguments(),
        }
    try:
        parser, ns = switch[origin]()
    except KeyError:
        log.exception("no calibration created for data origin: '{}'"
                      .format(origin))
        raise

    return parser, ns


def calibration_parameters(excess_args, origin, calib_help=False):
    """
    Obtain the calibration parameters.

    Parameters
    ----------
    excess_args : list
        List of arguments left over after intial parsing.
    origin : str
        Origin of data file e.g. hessio.
    calib_help : bool
        Print help message for calibration arguments.

    Returns
    -------
    params : dict
        Calibration parameter dict.
    unknown_args : list
        List of leftover cmdline arguments after parsing for calibration
        arguments.
    """

    parser, ns = calibration_parser(origin)

    if calib_help:
        parser.print_help()
        parser.exit()

    args, unknown_args = parser.parse_known_args(excess_args, ns)

    params = vars(args)
    for key, value in params.items():
        log.info("[{}] {}".format(key, value))

    return params, unknown_args


def calibrate_event(event, params, geom_dict=None):
    """
    Generic calibrator for events. Calls the calibrator corresponding to the
    source of the event, and stores the dl1 (calibrated_image) information into a
    new event container.

    Parameters
    ----------
    event : container
        A `ctapipe` event container
    params : dict
        REQUIRED:

        params['integrator'] - Integration scheme

        params['integration_window'] - Integration window size and shift of
        integration window centre

        (adapted such that window fits into readout).

        OPTIONAL:

        params['integration_clip_amp'] - Amplitude in p.e. above which the
        signal is clipped.

        params['integration_calib_scale'] : Identical to global variable
        CALIB_SCALE in reconstruct.c in hessioxxx software package. 0.92 is
        the default value (corresponds to HESS). The required value changes
        between cameras (GCT = 1.05).

        params['integration_sigamp'] - Amplitude in ADC counts above pedestal
        at which a signal is considered as significant (separate for
        high gain/low gain).
    geom_dict : dict
        Dict of pixel geometry for each telescope. Leave as None for automatic
        calculation when it is required.
        dict[(num_pixels, focal_length)] = `ctapipe.io.CameraGeometry`

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
            partial(mc.calibrate_mc, event=event, params=params)
        }
    try:
        calibrator = switch[event.meta['source']]
    except KeyError:
        log.exception("no calibration created for data origin: '{}'"
                      .format(event.meta['source']))
        raise

    # KPK: should not copy the event here! there is no reason to
    # Copying is
    # up to the user if they want to do it, not in the algorithms.
    #    calibrated = copy(event)

    # params stored in metadata
    event.dl1.meta.update(params)

    # Fill dl1
    event.dl1.reset()
    for telid in event.dl0.tels_with_data:
        nchan = event.inst.num_channels[telid]
        npix = event.inst.num_pixels[telid]
        event.dl1.tel[telid] = CalibratedCameraContainer()

        # Get geometry
        int_dict, inverted = integrator_dict()
        geom = None

        # Check if geom is even needed for integrator
        if inverted[params['integrator']] in integrators_requiring_geom():
            if geom_dict is not None and telid in geom_dict:
                geom = geom_dict[telid]
            else:
                log.debug("[calib] Guessing camera geometry")
                geom = CameraGeometry.guess(*event.inst.pixel_pos[telid],
                                            event.inst.optical_foclen[telid])
                log.debug("[calib] Camera geometry found")
                if geom_dict is not None:
                    geom_dict[telid] = geom

        pe, window, data_ped, peakpos = calibrator(telid=telid, geom=geom)
        tel = event.dl1.tel[telid]
        tel.calibrated_image = pe
        tel.peakpos = peakpos
        for chan in range(nchan):
            tel.integration_window[chan] = window[chan]
            tel.pedestal_subtracted_adc[chan] = data_ped[chan]

    return event


def calibrate_source(source, params, geom_dict=None):
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

        params['integration_window'] - Integration window size and shift of
        integration window centre

        (adapted such that window fits into readout).

        OPTIONAL:

        params['integration_clip_amp'] - Amplitude in p.e. above which the
        signal is clipped.

        params['integration_calib_scale'] : Identical to global variable
        CALIB_SCALE in reconstruct.c in hessioxxx software package. 0.92 is
        the default value (corresponds to HESS). The required value changes
        between cameras (GCT = 1.05).

        params['integration_sigamp'] - Amplitude in ADC counts above pedestal
        at which a signal is considered as significant (separate for
        high gain/low gain).
    geom_dict : dict
        Dict of pixel geometry for each telescope. Leave as None for automatic
        calculation when it is required. Can be used to only calculate a geom
        once per telescope by utilising a dicts mutability.
        dict[(num_pixels, focal_length)] = `ctapipe.io.CameraGeometry`

    Returns
    -------
    calibrated : container
        A new `ctapipe` event container containing the dl1 information, and a
        reference to all other information contained in the original event
        container.
    """
    if geom_dict is None:
        geom_dict = {}

    log.info("[calib] Calibration generator appended to source")
    for event in source:
        calibrate_event(event, params, geom_dict)
        yield event
