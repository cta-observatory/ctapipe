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

import numpy as np
from ctapipe.io import CameraGeometry
import logging
from scipy import interp
from .integrators import integrator_switch

logger = logging.getLogger(__name__)


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

        params['window'] - Integration window size

        params['shift'] - Starting sample for this integration

        (adapted such that window fits into readout).

    Returns
    -------
    correction : float
        Value of the integration correction for this instrument \n
    Returns None if params dict does not include all required parameters
    """
    if 'window' not in params or 'shift' not in params:
        return None

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
    start = ipeak - params['shift']
    window = params['window']
    if window > nsamples:
        window = nsamples
    if start < 0:
        start = 0
    if start + window > nsamples:
        start = nsamples - window

    correction = round((sum(y) * refstep) / (sum(y1[start:start + window]) *
                                             time_slice), 7)
    return correction


def calibrate_amplitude_mc(event, charge, telid, params, calib_scale=0.92):
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

        params['clip_amp'] - Amplitude in p.e. above which the signal is
        clipped.
    calib_scale : float
        Identical to global variable CALIB_SCALE in reconstruct.c in hessioxxx
        software package. 0.92 is the default value (corresponds to HESS). The
        required value changes between cameras (GCT = 1.05).

    Returns
    -------
    pe : ndarray
        array of pixels with integrated charge [photo-electrons]
        (pedestal substracted)
    """

    if charge is None:
        return None

    calib = event.dl0.tel[telid].calibration

    pe = charge * calib
    # TODO: add clever calib for prod3 and LG channel

    if "climp_amp" in params and params["clip_amp"] > 0:
        pe[np.where(pe > params["clip_amp"])] = params["clip_amp"]

    """
    pe_pix is in units of 'mean photo-electrons'
    (unit = mean p.e. signal.).
    We convert to experimentalist's 'peak photo-electrons'
    now (unit = most probable p.e. signal after experimental resolution).
    Keep in mind: peak(10 p.e.) != 10*peak(1 p.e.)
    """
    pe *= calib_scale
    # TODO: create dict of CALIB_SCALE for every instrument

    return pe


def integration_mc(event, telid, params):
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

        params['window'] - Integration window size

        params['shift'] - Starting sample for this integration

        (adapted such that window fits into readout).

    Returns
    -------
    charge : ndarray
        array of pixels with integrated charge [ADC counts]
        (pedestal substracted)
    integration_window : ndarray
        bool array of same shape as data. Specified which samples are included
        in the integration window

    Returns None if params dict does not include all required parameters
    """

    if event is None or telid < 0:
        return None
    if 'window' not in params or 'shift' not in params:
        return None

    # Obtain the data
    nsamples = event.dl0.tel[telid].num_samples
    data = np.array(list(event.dl0.tel[telid].adc_samples.values()))
    ped = event.dl0.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped/nsamples)
    geom = CameraGeometry.guess(*event.meta.pixel_pos[telid],
                                event.meta.optical_foclen[telid])

    # Integrate
    integration, integration_window = integrator_switch(data_ped, geom, params)

    # Get the integration correction
    int_corr = set_integration_correction(event, telid, params)
    window = sum(integration_window[0][0])
    if window == nsamples:
        int_corr = 1

    # Convert integration into charge
    charge = np.round(integration * int_corr).astype(np.int16, copy=False)

    return charge, integration_window
