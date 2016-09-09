"""
Module containing general functions that will calibrate any event regardless of
the source/telescope, and store the calibration inside the event container.
"""


from copy import copy
from .mc import calibrate_mc
from .integrators import integrator_dict, integrators_requiring_geom
from functools import partial
from ctapipe.io.containers import RawData, CalibratedCameraData
from ctapipe.io import CameraGeometry
from astropy import log


def calibration_arguments(parser):
    """
    Add the arguments for the calibration function to your argparser.

    Parameters
    ----------
    parser : `astropy.utils.compat.argparse.ArgumentParser`
    """
    integrators = ""
    int_dict, inverted = integrator_dict()
    for key, value in int_dict.items():
        integrators += " - {} = {}\n".format(key, value)

    parser.add_argument('--integrator', dest='integrator', action='store',
                        default=5, type=int,
                        help='which integration scheme should be used to '
                             'extract the charge? (default = 5)'
                             '\n{}'.format(integrators))
    parser.add_argument('--integration-window', dest='integration_window',
                        action='store', default=[7, 3], nargs=2, type=int,
                        help='Set integration window width and offset (to '
                             'before the peak) respectively, '
                             'e.g. --integration-window 7 3 (default)')
    parser.add_argument('--integration-sigamp', dest='integration_sigamp',
                        action='store', nargs='+', type=int, default=[2, 4],
                        help='Amplitude in ADC counts above pedestal at which '
                             'a signal is considered as significant, and used '
                             'for peak finding. '
                             '(separate for high gain/low gain), '
                             'e.g. --integration-sigamp 2 4 (default)')
    parser.add_argument('--integration-clip_amp', dest='integration_clip_amp',
                        action='store', type=int, default=None,
                        help='Amplitude in p.e. above which the signal is '
                             'clipped.')
    parser.add_argument('--integration-lwt', dest='integration_lwt',
                        action='store', type=int, default=0,
                        help='Weight of the local pixel (0: peak from '
                             'neighbours only, 1: local pixel counts as much '
                             'as any neighbour) default=0')
    parser.add_argument('--integration-calib_scale',
                        dest='integration_calib_scale',
                        action='store', type=float, default=0.92,
                        help='Used for conversion from ADC to pe. Identical '
                             'to global variable CALIB_SCALE in '
                             'reconstruct.c in hessioxxx software package. '
                             '0.92 is the default value (corresponds to '
                             'HESS). The required value changes between '
                             'cameras (GCT = 1.05).')


def calibration_parameters(args):
    """
    Parse you argpasers arguments for calibration parameters, and store them
    inside a dict.

    Parameters
    ----------
    args : `astropy.utils.compat.argparse.ArgumentParser.parse_args()`

    Returns
    -------
    parameters : dict
        dictionary containing the formatted calibration parameters
    """
    parameters = {}
    if args.integrator is not None:
        integrator_names, inverse = integrator_dict()
        try:
            parameters['integrator'] = integrator_names[args.integrator]
        except KeyError:
            log.exception('[calib] Specified integrator does not exist: {}'
                          .format(args.integrator))
            raise
    if args.integration_window is not None:
        parameters['window'] = args.integration_window[0]
        parameters['shift'] = args.integration_window[1]
    if args.integration_sigamp is not None:
        parameters['sigamp'] = args.integration_sigamp[:2]
    if args.integration_clip_amp is not None:
        parameters['clip_amp'] = args.integration_clip_amp
    if args.integration_lwt is not None:
        parameters['lwt'] = args.integration_lwt
    if args.integration_calib_scale is not None:
        parameters['calib_scale'] = args.integration_calib_scale

    for key, value in parameters.items():
        log.info("[{}] {}".format(key, value))

    return parameters


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
            partial(calibrate_mc, event=event, params=params)
        }
    try:
        calibrator = switch[event.meta.source]
    except KeyError:
        log.exception("unknown event source '{}'".format(event.meta.source))
        raise

    calibrated = copy(event)

    # Add dl1 to the event container (if it hasn't already been added)
    try:
        calibrated.add_item("dl1", RawData())
        calibrated.dl1.run_id = event.dl0.run_id
        calibrated.dl1.event_id = event.dl0.event_id
        calibrated.dl1.tels_with_data = event.dl0.tels_with_data
        calibrated.dl1.calibration_parameters = params
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

        # Get geometry
        int_dict, inverted = integrator_dict()
        geom = None
        cam_dimensions = (event.dl0.tel[telid].num_pixels,
                          event.meta.optical_foclen[telid])
        # Check if geom is even needed for integrator
        if inverted[params['integrator']] in integrators_requiring_geom():
            if geom_dict is not None and cam_dimensions in geom_dict:
                geom = geom_dict[cam_dimensions]
            else:
                log.debug("[calib] Guessing camera geometry")
                geom = CameraGeometry.guess(*event.meta.pixel_pos[telid],
                                            event.meta.optical_foclen[telid])
                log.debug("[calib] Camera geometry found")
                if geom_dict is not None:
                    geom_dict[cam_dimensions] = geom

        pe, window, data_ped, peakpos = calibrator(telid=telid, geom=geom)
        calibrated.dl1.tel[telid].pe_charge = pe
        calibrated.dl1.tel[telid].peakpos = peakpos
        for chan in range(nchan):
            calibrated.dl1.tel[telid].integration_window[chan] = window[chan]
            calibrated.dl1.tel[telid].pedestal_subtracted_adc[chan] = \
                data_ped[chan]

    return calibrated


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
        calibrated = calibrate_event(event, params, geom_dict)

        yield calibrated
