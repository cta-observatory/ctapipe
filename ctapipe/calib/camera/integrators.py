""" Contains the common integrators that are called by the individual
calibrators to extract charge from a waveform.

These methods have no prior known information other than the 3 dimensional
pedestal-subtracted data array containing every sample, for every pixel,
in every channel. Also may require the geometry of the camera pixels.

In general the integration functions corresponds one to one in name and
functionality with those in hessioxxx package.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def integrator_switch(data, geom, params):
    """
    Integrator switch using params['integrator'] to dictate which integration
    is applied.

    Parameters
    ----------
    data : ndarray
        3 dimensional numpy array containing the pedestal subtracted adc
        counts. data[nchan][npix][nsamples]
    geom : `ctapipe.io.CameraGeometry`
        geometry of the camera's pixels
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

    if data is None:
        return None
    if 'window' not in params or 'shift' not in params:
        return None

    switch = {
        'full_integration':
            lambda: full_integration(data),
        'simple_integration':
            lambda: simple_integration(data, params),
        'global_peak_integration':
            lambda: global_peak_integration(data, params),
        'local_peak_integration':
            lambda: local_peak_integration(data, params),
        'nb_peak_integration':
            lambda: nb_peak_integration(data, geom, params),
        }
    try:
        integrator = switch[params['integrator']]()
    except KeyError:
        integrator = switch[None]()

    return integrator


def full_integration(data):
    """
    Integrate full readout traces.

    No weighting of individual samples is applied.

    Parameters
    ----------
    data : ndarray
        3 dimensional numpy array containing the pedestal subtracted adc
        counts. data[nchan][npix][nsamples]

    Returns
    -------
    charge : array
        array of pixels with integrated charge [ADC counts]
        (pedestal substracted)
    """

    if data is None:
        return None

    integration_window = np.ones_like(data, dtype=bool)
    integration = data.sum(2)

    return integration, integration_window


def simple_integration(data, params):
    """
    Integrate sample-mode data (traces) over a common and fixed interval.

    The integration window can be anywhere in the available length of
    the traces.

    No weighting of individual samples is applied.

    Note: for multiple gains, this results in identical integration regions.

    Parameters
    ----------
    data : ndarray
        3 dimensional numpy array containing the pedestal subtracted adc
        counts. data[nchan][npix][nsamples]
    params : dict
        REQUIRED:

        params['window'] - Integration window size

        params['shift'] - Starting sample for this integration

        (adapted such that window fits into readout).

    Returns
    -------
    integration : ndarray
        array of pixels with integrated charge [ADC counts]
        (pedestal substracted)
    integration_window : ndarray
        bool array of same shape as data. Specified which samples are included
        in the integration window

    Returns None if params dict does not include all required parameters
    """

    if data is None:
        return None
    if 'window' not in params or 'shift' not in params:
        return None

    nchan, npix, nsamples = data.shape

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

    # Select entries
    integration_window = np.zeros_like(data, dtype=bool)
    integration_window[:, :, start:start + window] = True
    data = data * integration_window

    # Integrate
    integration = data.sum(2)

    return integration, integration_window


def global_peak_integration(data, params):
    """
    Integrate sample-mode data (traces) over a common interval around a
    global signal peak.

    The integration window can be anywhere in the available length of the
    traces.

    No weighting of individual samples is applied.

    Parameters
    ----------
    data : ndarray
        3 dimensional numpy array containing the pedestal subtracted adc
        counts. data[nchan][npix][nsamples]
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
    integration : ndarray
        array of pixels with integrated charge [ADC counts]
        (pedestal substracted)
    integration_window : ndarray
        bool array of same shape as data. Specified which samples are included
        in the integration window

    Returns None if params dict does not include all required parameters
    """

    if data is None:
        return None
    if 'window' not in params or 'shift' not in params:
        return None

    nchan, npix, nsamples = data.shape

    # Extract significant entries
    sig_entries = np.ones_like(data, dtype=bool)
    if 'sigamp' in params:
        sigamp_cut = params['sigamp']
        for i in range(len(sigamp_cut) if len(sigamp_cut) <= nchan else nchan):
            sig_entries[i] = data[i] > sigamp_cut[i]
    sig_pixels = np.any(sig_entries, axis=2)
    sig_channel = np.any(sig_pixels, axis=1)
    if not sig_channel[0] == True:
        logger.error("[ERROR] sigamp value excludes all values in HG channel")
    significant_data = data * sig_entries

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

    # Select entries
    integration_window = np.zeros_like(data, dtype=bool)
    for i in range(nchan):
        integration_window[i, :, start[i]:start[i] + window] = True
    data = data * integration_window

    # Integrate
    integration = data.sum(2)

    return integration, integration_window


def local_peak_integration(data, params):
    """
    Integrate sample-mode data (traces) around a pixel-local signal peak.

    The integration window can be anywhere in the available
    length of the traces.

    No weighting of individual samples is applied.

    Parameters
    ----------
    data : ndarray
        3 dimensional numpy array containing the pedestal subtracted adc
        counts. data[nchan][npix][nsamples]
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
    integration : ndarray
        array of pixels with integrated charge [ADC counts]
        (pedestal substracted)
    integration_window : ndarray
        bool array of same shape as data. Specified which samples are included
        in the integration window

    Returns None if params dict does not include all required parameters
    """

    if data is None:
        return None
    if 'window' not in params or 'shift' not in params:
        return None

    nchan, npix, nsamples = data.shape

    # Extract significant entries
    sig_entries = np.ones_like(data, dtype=bool)
    if 'sigamp' in params:
        sigamp_cut = params['sigamp']
        for i in range(len(sigamp_cut) if len(sigamp_cut) <= nchan else nchan):
            sig_entries[i] = data[i] > sigamp_cut[i]
    sig_pixels = np.any(sig_entries, axis=2)
    sig_channel = np.any(sig_pixels, axis=1)
    if not sig_channel[0] == True:
        logger.error("[ERROR] sigamp value excludes all values in HG channel")
    significant_data = data * sig_entries

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

    # Select entries
    integration_window = np.zeros_like(data, dtype=bool)
    for i in range(nchan):
        for j in range(npix):
            integration_window[i, j, start[i, j]:start[i, j] + window] = True
    data = data * integration_window

    # Integrate
    integration = data.sum(2)

    return integration, integration_window


def nb_peak_integration(data, geom, params):
    """
    Integrate sample-mode data (traces) around a peak in the signal sum of
    neighbouring pixels.

    The integration window can be anywhere in the available length
    of the traces.

    No weighting of individual samples is applied.

    Parameters
    ----------
    data : ndarray
        3 dimensional numpy array containing the pedestal subtracted adc
        counts. data[nchan][npix][nsamples]
    geom : `ctapipe.io.CameraGeometry`
        geometry of the camera's pixels
    params : dict
        REQUIRED:

        params['window'] - Integration window size

        params['shift'] - Starting sample for this integration

        (adapted such that window fits into readout).

        OPTIONAL:

        params['sigamp'] - Amplitude in ADC counts above pedestal at which a
        signal is considered as significant (separate for high gain/low gain).

        params['lwt']
        - Weight of the local pixel (0: peak from neighbours
        only, 1: local pixel counts as much as any neighbour).

    Returns
    -------
    integration : ndarray
        array of pixels with integrated charge [ADC counts]
        (pedestal substracted)
    integration_window : ndarray
        bool array of same shape as data. Specified which samples are included
        in the integration window

    Returns None if params dict does not include all required parameters
    """

    if data is None:
        return None
    if 'window' not in params or 'shift' not in params:
        return None

    nchan, npix, nsamples = data.shape

    # Extract significant entries
    sig_entries = np.ones_like(data, dtype=bool)
    if 'sigamp' in params:
        sigamp_cut = params['sigamp']
        for i in range(len(sigamp_cut) if len(sigamp_cut) <= nchan else nchan):
            sig_entries[i] = data[i] > sigamp_cut[i]
    sig_pixels = np.any(sig_entries, axis=2)
    sig_channel = np.any(sig_pixels, axis=1)
    if not sig_channel[0] == True:
        logger.error("[ERROR] sigamp value excludes all values in HG channel")
    significant_data = data * sig_entries

    # Define window
    lwt = 0 if 'lwt' not in params else params['lwt']
    neighbour_list = geom.neighbors
    peakpos = np.zeros_like(sig_pixels, dtype=np.int16)
    for ipix, neighbours in enumerate(neighbour_list):
        nb_data = significant_data[:, neighbours]
        weighted_pixel = significant_data[:, ipix] * lwt
        """@type weighted_pixel: numpy.core.multiarray.ndarray"""
        pixel_expanded = np.expand_dims(weighted_pixel, axis=1)
        all_data = np.concatenate((nb_data, pixel_expanded), axis=1)
        sum_data = all_data.sum(1)
        peakpos[:, ipix] = sum_data.argmax(1)
    start = (peakpos - params['shift']).astype(np.int16, copy=False)
    window = params['window']

    # Check window is within readout
    if window > nsamples:
        window = nsamples
    start[np.where(start < 0)] = 0
    start[np.where(start + window > nsamples)] = nsamples - window

    # Select entries
    integration_window = np.zeros_like(data, dtype=bool)
    for i in range(nchan):
        for j in range(npix):
            integration_window[i, j, start[i, j]:start[i, j] + window] = True
    data = data * integration_window

    # Integrate
    integration = data.sum(2)

    return integration, integration_window
