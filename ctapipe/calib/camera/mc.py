"""
Integrate sample-mode data (traces) Functions
and convert the integral pixel ADC count to photo-electrons
"""

import numpy as np
# from pyhessio import get_num_channel, get_ref_shapes, get_ref_step,\
#     get_lrefshape, get_time_slice, get_num_samples, \
#     get_adc_sample, get_num_pixels
from ctapipe import io
import logging
from scipy import interp

logger = logging.getLogger(__name__)

__all__ = [
    'set_integration_correction',
    'pixel_integration_mc',
    'full_integration_mc',
    'simple_integration_mc',
    'global_peak_integration_mc',
    'local_peak_integration_mc',
    'nb_peak_integration_mc',
    'calibrate_amplitude_mc'
]

# CALIB_SCALE = 0.92  # HESS Value
CALIB_SCALE = 1.05  # GCT Value
# TODO: create dict of CALIB_SCALE for every instrument

"""

The function in this module are the same to their corresponding ones in
- read_hess.c
- reconstruct.c
in hessioxxx software package, just in some caes the suffix "_mc" has
been added here to the original name function.

Note: Input MC version = prod2. For future MC versions the calibration
function might be different for each camera type.
It has not been tested so far.

In general the integration functions corresponds one to one in name and
functionality with those in hessioxxx package.
The same the interpolation of the pulse shape and the adc2pe conversion.

"""


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


def pixel_integration_mc(event, telid, params):
    """
    Integrator for mc files

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

        params['integrator'] - pixel integration algorithm
            - "full_integration": full digitized range integrated
            amplitude-pedestal
            - "simple_integration": fixed integration region (window)
            - "global_peak_integration": integration region by global peak of
            significant pixels
            - "local_peak_integration": peak in each pixel determined
            independently
            - "nb_peak_integration": peak postion found by summin neighbours

    Returns
    -------
    integrator : lambda
        function corresponding to the specified integrator
    Returns None if params dict does not include all required params
    """

    if event is None or telid < 0:
        return None
    if 'window' not in params or 'shift' not in params:
        return None

    switch = {
        'full_integration':
            lambda: full_integration_mc(event, telid),
        'simple_integration':
            lambda: simple_integration_mc(event, telid, params),
        'global_peak_integration':
            lambda: global_peak_integration_mc(event, telid, params),
        'local_peak_integration':
            lambda: local_peak_integration_mc(event, telid, params),
        'nb_peak_integration':
            lambda: nb_peak_integration_mc(event, telid, params),
        }
    try:
        integrator = switch[params['integrator']]()
    except KeyError:
        integrator = switch[None]()

    return integrator


def full_integration_mc(event, telid):
    """
    Use full digitized range for the integration amplitude
    algorithm (sum - pedestal)

    No weighting of individual samples is applied.

    Parameters
    ----------
    event : container
        A `ctapipe` event container
    telid : int
        telescope id

    Returns
    -------
    charge : array
        array of pixels with integrated charge [ADC counts]
        (pedestal substracted)
    """

    if event is None or telid < 0:
        return None

    data = np.array(list(event.dl0.tel[telid].adc_samples.values()))
    ped = event.dl0.tel[telid].pedestal
    integration = data.sum(2)
    charge = np.round(integration - ped).astype(np.int16, copy=False)

    return charge


def simple_integration_mc(event, telid, params):
    """
    Integrate sample-mode data (traces) over a common and fixed interval.

    The integration window can be anywhere in the available length of
    the traces.
    Since the calibration function subtracts a pedestal that corresponds to the
    total length of the traces we may also have to add a pedestal contribution
    for the samples not summed up.
    No weighting of individual samples is applied.
    Note: for multiple gains, this results in identical integration regions.

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
    charge : array
        array of pixels with integrated charge [ADC counts]
        (pedestal substracted)
    Returns None if params dict does not include all required parameters
    """

    if event is None or telid < 0:
        return None
    if 'window' not in params or 'shift' not in params:
        return None

    # Obtain the data
    nsamples = event.dl0.tel[telid].num_samples
    nchan = event.dl0.tel[telid].num_channels
    data = np.array(list(event.dl0.tel[telid].adc_samples.values()))
    ped = event.dl0.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped/nsamples)

    # Define window
    window = params['window']
    start = np.array([params['shift']], dtype=np.int16)

    # Check window is within readout
    if window > nsamples:
        window = nsamples
    if start < 0:
        start = 0
    if start + window > nsamples:
        start = nsamples - window

    # Get the integration correction
    int_corr = set_integration_correction(event, telid, params)
    if window == nsamples:
        int_corr = 1

    # Select entries
    data_window = np.zeros_like(data_ped, dtype=bool)
    for i in range(nchan):
        data_window[i, :, start[i]:start[i] + window] = True
    data_ped = data_ped * data_window

    # Integrate
    integration = data_ped.sum(2)
    charge = np.round(integration * int_corr).astype(np.int16, copy=False)

    return charge


def global_peak_integration_mc(event, telid, params):
    """
    Integrate sample-mode data (traces) over a common interval around a
    global signal peak.

    The integration window can be anywhere in the available length of the
    traces.
    No weighting of individual samples is applied.

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

        OPTIONAL:

        params['sigamp'] - Amplitude in ADC counts above pedestal at which a
        signal is considered as significant (separate for high gain/low gain).

    Returns
    -------
    charge : array
        array of pixels with integrated charge [ADC counts]
        (pedestal substracted)
    Returns None if params dict does not include all required parameters
    """

    if event is None or telid < 0:
        return None
    if 'window' not in params or 'shift' not in params:
        return None

    # Obtain the data
    nchan = event.dl0.tel[telid].num_channels
    nsamples = event.dl0.tel[telid].num_samples
    data = np.array(list(event.dl0.tel[telid].adc_samples.values()))
    ped = event.dl0.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped/nsamples)

    # Extract significant entries
    sig_entries = np.ones_like(data_ped, dtype=bool)
    if 'sigamp' in params:
        sigamp_cut = params['sigamp']
        for i in range(len(sigamp_cut) if len(sigamp_cut) <= nchan else nchan):
            sig_entries[i] = data_ped[i] > sigamp_cut[i]
    sig_pixels = np.any(sig_entries, axis=2)
    sig_channel = np.any(sig_pixels, axis=1)
    if not sig_channel[0] == True:
        logger.error("[ERROR] sigamp value excludes all values in HG channel")
    significant_data = data_ped * sig_entries

    # Define window
    max_time = significant_data.argmax(2)
    max_sample = significant_data.max(2)
    max_time_sample_sum = (max_time * max_sample).sum(1)
    max_sample_sum = max_sample.sum(1)
    peakpos = np.zeros_like(max_sample_sum, dtype=np.int16)
    if 0 not in max_sample_sum:
        peakpos = np.round(max_time_sample_sum / max_sample_sum)
    else:  # If the LG is not significant, takes the HG peakpos
        peakpos[0] = np.round(max_time_sample_sum[0] / max_sample_sum[0])
        peakpos[1] = peakpos[0]
        peakpos = peakpos
    start = (peakpos - params['shift']).astype(np.int16, copy=False)
    window = params['window']

    # Check window is within readout
    if window > nsamples:
        window = nsamples
    start[np.where(start < 0)] = 0
    start[np.where(start + window > nsamples)] = nsamples - window

    # Get the integration correction
    int_corr = set_integration_correction(event, telid, params)
    if window == nsamples:
        int_corr = 1

    # Select entries
    data_window = np.zeros_like(data_ped, dtype=bool)
    for i in range(nchan):
        data_window[i, :, start[i]:start[i] + window] = True
    data_ped = data_ped * data_window

    # Integrate
    integration = data_ped.sum(2)
    charge = np.round(integration * int_corr).astype(np.int16, copy=False)

    return charge


def local_peak_integration_mc(event, telid, params):
    """
    Integrate sample-mode data (traces) around a pixel-local signal peak.

    The integration window can be anywhere in the available
    length of the traces.
    No weighting of individual samples is applied.

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

        OPTIONAL:

        params['sigamp'] - Amplitude in ADC counts above pedestal at which a
        signal is considered as significant (separate for high gain/low gain).

    Returns
    -------
    charge : array
        array of pixels with integrated charge [ADC counts]
        (pedestal substracted)
    Returns None if params dict does not include all required parameters
    """

    if event is None or telid < 0:
        return None
    if 'window' not in params or 'shift' not in params:
        return None

    # Obtain the data
    nchan = event.dl0.tel[telid].num_channels
    nsamples = event.dl0.tel[telid].num_samples
    npix = event.dl0.tel[telid].num_pixels
    data = np.array(list(event.dl0.tel[telid].adc_samples.values()))
    ped = event.dl0.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped/nsamples)

    # Extract significant entries
    sig_entries = np.ones_like(data_ped, dtype=bool)
    if 'sigamp' in params:
        sigamp_cut = params['sigamp']
        for i in range(len(sigamp_cut) if len(sigamp_cut) <= nchan else nchan):
            sig_entries[i] = data_ped[i] > sigamp_cut[i]
    sig_pixels = np.any(sig_entries, axis=2)
    sig_channel = np.any(sig_pixels, axis=1)
    if not sig_channel[0] == True:
        logger.error("[ERROR] sigamp value excludes all values in HG channel")
    significant_data = data_ped * sig_entries

    # Define window
    peakpos = significant_data.argmax(2)
    if nchan > 1:  # If the LG is not significant, takes the HG peakpos
        peakpos[1] = np.where(sig_pixels[1] < sig_pixels[0], peakpos[0],
                              peakpos[1])
    start = (peakpos - params['shift']).astype(np.int16, copy=False)
    window = params['window']

    # Check window is within readout
    if window > nsamples:
        window = nsamples
    start[np.where(start < 0)] = 0
    start[np.where(start + window > nsamples)] = nsamples - window

    # Get the integration correction
    int_corr = set_integration_correction(event, telid, params)
    if window == nsamples:
        int_corr = 1

    # Select entries
    data_window = np.zeros_like(data_ped, dtype=bool)
    for i in range(nchan):
        for j in range(npix):
            data_window[i, :, start[i, j]:start[i, j] + window] = True
    data_ped = data_ped * data_window

    # Integrate
    integration = data_ped.sum(2)
    charge = np.round(integration * int_corr).astype(np.int16, copy=False)

    return charge


def nb_peak_integration_mc(event, telid, params):
    """
    Integrate sample-mode data (traces) around a peak in the signal sum of
    neighbouring pixels.

    The integration window can be anywhere in the available length
    of the traces.
    No weighting of individual samples is applied.

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

        OPTIONAL:

        params['sigamp'] - Amplitude in ADC counts above pedestal at which a
        signal is considered as significant (separate for high gain/low gain).

        params['lwt'] - Weight of the local pixel (0: peak from neighbours
        only, 1: local pixel counts as much as any neighbour).

    Returns
    -------
    charge : array
        array of pixels with integrated charge [ADC counts]
        (pedestal substracted)
    Returns None if params dict does not include all required parameters
    """

    if event is None or telid < 0:
        return None
    if 'window' not in params or 'shift' not in params:
        return None

    # Obtain the data
    nchan = event.dl0.tel[telid].num_channels
    nsamples = event.dl0.tel[telid].num_samples
    npix = event.dl0.tel[telid].num_pixels
    data = np.array(list(event.dl0.tel[telid].adc_samples.values()))
    ped = event.dl0.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped/nsamples)

    # Extract significant entries
    sig_entries = np.ones_like(data_ped, dtype=bool)
    if 'sigamp' in params:
        sigamp_cut = params['sigamp']
        for i in range(len(sigamp_cut) if len(sigamp_cut) <= nchan else nchan):
            sig_entries[i] = data_ped[i] > sigamp_cut[i]
    sig_pixels = np.any(sig_entries, axis=2)
    sig_channel = np.any(sig_pixels, axis=1)
    if not sig_channel[0] == True:
        logger.error("[ERROR] sigamp value excludes all values in HG channel")
    significant_data = data_ped * sig_entries

    # Define window
    lwt = params['lwt']
    geom = io.CameraGeometry.guess(*event.meta.pixel_pos[telid],
                                   event.meta.optical_foclen[telid])
    neighbour_list = geom.neighbors
    peakpos = np.zeros_like(sig_pixels, dtype=np.int16)
    for ipix, neighbours in enumerate(neighbour_list):
        nb_data = significant_data[:, neighbours]
        pixel = np.expand_dims(lwt*significant_data[:, ipix], axis=1)
        all_data = np.concatenate((nb_data, pixel), axis=1)
        sum_data = all_data.sum(1)
        peakpos[:, ipix] = sum_data.argmax(1)
    start = (peakpos - params['shift']).astype(np.int16, copy=False)
    window = params['window']

    # Check window is within readout
    if window > nsamples:
        window = nsamples
    start[np.where(start < 0)] = 0
    start[np.where(start + window > nsamples)] = nsamples - window

    # Get the integration correction
    int_corr = set_integration_correction(event, telid, params)
    if window == nsamples:
        int_corr = 1

    # Select entries
    data_window = np.zeros_like(data_ped, dtype=bool)
    for i in range(nchan):
        for j in range(npix):
            data_window[i, :, start[i, j]:start[i, j] + window] = True
    data_ped = data_ped * data_window

    # Integrate
    integration = data_ped.sum(2)
    charge = np.round(integration * int_corr).astype(np.int16, copy=False)

    return charge


def calibrate_amplitude_mc(event, charge, telid, params):
    """
    Convert charge from ADC counts to photo-electrons

    Parameters
    ----------
    event : container
        A `ctapipe` event container
    charge : int
        array of pixels with integrated charge [ADC counts]
        (pedestal substracted)
    telid : int
        telescope id
    params : dict
        OPTIONAL:

        params['clip_amp'] - Amplitude in p.e. above which the signal is
        clipped.

    Returns
    -------
    pe : array
        array of pixels with integrated charge [photo-electrons]
        (pedestal substracted)
    """

    if charge is None:
        return None

    calib = event.dl0.tel[telid].calibration

    pe = np.array(charge * calib)
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
    pe *= CALIB_SCALE

    return pe
