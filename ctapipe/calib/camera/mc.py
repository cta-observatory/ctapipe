"""
Calibration for MC (simtelarray) files.

The interpolation of the pulse shape and the adc2pe conversion in this module
are the same to their corresponding ones in
- read_hess.c
- reconstruct.c
in hessioxxx software package.

Note: Input MC version = prod2. For future MC versions the calibration
function might be different for each camera type.
"""
import argparse

import numpy as np
from ctapipe.io import CameraGeometry
from astropy import log
from scipy import interp
from .integrators import integrator_switch, integrators_requiring_geom, \
    integrator_dict


def calibration_arguments():
    """
    Obtain an argparser with the arguments for the MC calibration.

    Returns
    -------
    parser
        MC calibration argparser
    ns : `argparse.Namespace`
        Namespace containing the correction for default values so they use
        a custom Action
    """

    integrators = ""
    int_dict, inverted = integrator_dict()
    for key, value in int_dict.items():
        integrators += " - {} = {}\n".format(key, value)

    class IntegratorAction(argparse.Action):
        def __call__(self, parser0, namespace, values, option_string=None):
            setattr(namespace, self.dest, int_dict[values])

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    action = parser.add_argument('--integrator', dest='integrator',
                                 action=IntegratorAction,
                                 default=5, type=int, choices=int_dict.keys(),
                                 help='which integration scheme should be '
                                      'used to extract the '
                                      'charge? \n{}'.format(integrators))
    # Convert default using IntegratorAction
    ns = argparse.Namespace()
    action(parser, ns, action.default)

    parser.add_argument('--integration-window', dest='integration_window',
                        action='store', default=[7, 3], nargs=2, type=int,
                        help='Set integration window width and offset (to '
                             'before the peak) respectively, '
                             'e.g. --integration-window 7 3')
    parser.add_argument('--integration-sigamp', dest='integration_sigamp',
                        action='store', nargs='+', type=int, default=[2, 4],
                        help='Amplitude in ADC counts above pedestal at which '
                             'a signal is considered as significant, and used '
                             'for peak finding. '
                             '(separate for high gain/low gain), '
                             'e.g. --integration-sigamp 2 4')
    parser.add_argument('--integration-clip_amp', dest='integration_clip_amp',
                        action='store', type=int, default=None,
                        help='Amplitude in p.e. above which the signal is '
                             'clipped.')
    parser.add_argument('--integration-lwt', dest='integration_lwt',
                        action='store', type=int, default=0,
                        help='Weight of the local pixel (0: peak from '
                             'neighbours only, 1: local pixel counts as much '
                             'as any neighbour)')
    parser.add_argument('--integration-calib_scale',
                        dest='integration_calib_scale',
                        action='store', type=float, default=0.92,
                        help='Used for conversion from ADC to pe. Identical '
                             'to global variable CALIB_SCALE in '
                             'reconstruct.c in hessioxxx software package. '
                             'The required value changes between '
                             'cameras (HESS = 0.92, GCT = 1.05).')

    return parser, ns


def set_integration_correction(event, telid, params):
    """
    Obtain the integration correction for the window specified

    Parameters
    ----------
    event : container
        A `ctapipe` event container
    telid : int
        telescope id
    params : dict
        REQUIRED:

        params['integration_window'] - Integration window size and shift of
        integration window centre

        (adapted such that window fits into readout).

    Returns
    -------
    correction : float
        Value of the integration correction for this instrument \n
    """

    try:
        if 'integration_window' not in params:
            raise KeyError()
    except KeyError:
        log.exception("[ERROR] missing required params")
        raise

    nchan = event.dl0.tel[telid].num_channels
    nsamples = event.dl0.tel[telid].num_samples

    # Reference pulse parameters
    refshapes = np.array(list(event.mc.tel[telid].refshapes.values()))
    refstep = event.mc.tel[telid].refstep
    nrefstep = event.mc.tel[telid].lrefshape
    x = np.arange(0, refstep*nrefstep, refstep)
    y = refshapes[nchan-1]
    refipeak = np.argmax(y)

    # Sampling pulse parameters
    time_slice = event.mc.tel[telid].time_slice
    x1 = np.arange(0, refstep*nrefstep, time_slice)
    y1 = interp(x1, x, y)
    ipeak = np.argmin(np.abs(x1-x[refipeak]))

    # Check window is within readout
    window = params['integration_window'][0]
    shift = params['integration_window'][1]
    start = ipeak - shift
    if window > nsamples:
        window = nsamples
    if start < 0:
        start = 0
    if start + window > nsamples:
        start = nsamples - window

    correction = round((sum(y) * refstep) / (sum(y1[start:start + window]) *
                                             time_slice), 7)

    return correction


def calibrate_amplitude_mc(event, charge, telid, params):
    """
    Convert charge from ADC counts to photo-electrons

    Parameters
    ----------
    event : container
        A `ctapipe` event container
    charge : ndarray
        array of pixels with integrated charge [ADC counts]
        (pedestal substracted)
    telid : int
        telescope id
    params : dict
        OPTIONAL:

        params['integration_clip_amp'] - Amplitude in p.e. above which the
        signal is clipped.

        params['integration_calib_scale'] : Identical to global variable
        CALIB_SCALE in reconstruct.c in hessioxxx software package. 0.92 is
        the default value (corresponds to HESS). The required value changes
        between cameras (GCT = 1.05).

    Returns
    -------
    pe : ndarray
        array of pixels with integrated charge [photo-electrons]
        (pedestal substracted)
    """

    calib = event.dl0.tel[telid].calibration

    pe = charge * calib
    # TODO: add clever calib for prod3 and LG channel

    if "integration_clip_amp" in params and params["integration_clip_amp"]:
        pe[np.where(pe > params["integration_clip_amp"])] = \
            params["integration_clip_amp"]
    if "integration_calib_scale" not in params:
        # Store default value into mutable dict, so it is preserved
        params["integration_calib_scale"] = 0.92  # Correct value for HESS
        log.info("[calib] Default calib_scale set: {}"
                 .format(params["integration_calib_scale"]))
    calib_scale = params["integration_calib_scale"]

    """
    pe_pix is in units of 'mean photo-electrons'
    (unit = mean p.e. signal.).
    We convert to experimentalist's 'peak photo-electrons'
    now (unit = most probable p.e. signal after experimental resolution).
    Keep in mind: peak(10 p.e.) != 10*peak(1 p.e.)
    """
    scaled_pe = pe * calib_scale
    # TODO: create dict of CALIB_SCALE for every instrument

    return scaled_pe


def integration_mc(event, telid, params, geom=None):
    """
    Generic integrator for mc files. Calls the integrator_switch for actual
    integration. Subtracts the pedestal and applies the integration correction.

    Parameters
    ----------
    event : container
        A `ctapipe` event container
    telid : int
        telescope id
    params : dict
        REQUIRED:

        params['integrator'] - Integration scheme

        params['integration_window'] - Integration window size and shift of
        integration window centre

        (adapted such that window fits into readout).

        OPTIONAL:

        params['integration_sigamp'] - Amplitude in ADC counts above pedestal
        at which a signal is considered as significant (separate for
        high gain/low gain).
    geom : `ctapipe.io.CameraGeometry`
        geometry of the camera's pixels. Leave as None for automatic
        calculation when it is required.

    Returns
    -------
    charge : ndarray
        array of pixels with integrated charge [ADC counts]
        (pedestal substracted)
    integration_window : ndarray
        bool array of same shape as data. Specified which samples are included
        in the integration window
    data_ped : ndarray
        pedestal subtracted data
    peakpos : ndarray
        position of the peak as determined by the peak-finding algorithm
        for each pixel and channel
    """

    # Obtain the data
    nsamples = event.dl0.tel[telid].num_samples
    data = np.array(list(event.dl0.tel[telid].adc_samples.values()))
    ped = event.dl0.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped/nsamples)
    int_dict, inverted = integrator_dict()
    if geom is None and inverted[params['integrator']]\
            in integrators_requiring_geom():
        log.debug("[calib] Guessing camera geometry")
        geom = CameraGeometry.guess(*event.meta.pixel_pos[telid],
                                    event.meta.optical_foclen[telid])
        log.debug("[calib] Camera geometry found")

    # Integrate
    integration, integration_window, peakpos = \
        integrator_switch(data_ped, geom, params)

    # Get the integration correction
    int_corr = set_integration_correction(event, telid, params)
    if peakpos[0] is None:
        int_corr = 1

    # Convert integration into charge
    charge = np.round(integration * int_corr)

    return charge, integration_window, data_ped, peakpos


def calibrate_mc(event, telid, params, geom=None):
    """
    Generic calibrator for mc files. Calls the itegrator function to obtain
    the ADC charge, then calibrates that into photo-electrons.

    Parameters
    ----------
    event : container
        A `ctapipe` event container
    telid : int
        telescope id
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
    geom : `ctapipe.io.CameraGeometry`
        geometry of the camera's pixels. Leave as None for automatic
        calculation when it is required.

    Returns
    -------
    channel_pe : ndarray
        array of pixels with integrated charge [photo-electrons] for the
        chosen channel
        (pedestal substracted)
    window : ndarray
        bool array of same shape as data. Specified which samples are included
        in the integration window
    data_ped : ndarray
        pedestal subtracted data
    channel_peakpos : ndarray
        position of the peak as determined by the peak-finding algorithm
        for each pixel, and for the channel that has been chosen to be used.
    """

    charge, window, data_ped, peakpos = \
        integration_mc(event, telid, params, geom)
    pe = calibrate_amplitude_mc(event, charge, telid, params)
    if 'integration_clip_amp' in params and params['integration_clip_amp']:
        pe[np.where(pe > params['integration_clip_amp'])] \
            = params['integration_clip_amp']

    # Decide between HG and LG channel
    # TODO: add actual logic for decision, currently uses only HG
    channel_pe = pe[0]
    channel_peakpos = peakpos[0]

    return channel_pe, window, data_ped, channel_peakpos
