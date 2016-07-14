"""
Integrate sample-mode data (traces) Functions
and
Convert the integral pixel ADC count to photo-electrons
"""

import sys
import numpy as np
from numpy import round
from pyhessio import *
from ctapipe import io
from astropy import units as u
import logging
from scipy import interp, integrate, interpolate
import matplotlib
from pylab import *
import time

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

# CALIB_SCALE = 0.92 # HESS Value
CALIB_SCALE = 1.05 # GCT Value
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


def set_integration_correction(telid, params):
    """
    Obtain the integration correction for the window specified

    Parameters
    ----------
    telid : int
        telescope id
    params : dict
        parameters['shift']  Shift before the peak for window start
        parameters['window']  Integration window size
        (adapted such that window fits into readout).

    Returns
    -------
    correction : float
        Value of the integration correction for this instrument
        Returns None if params dict does not include all required parameters
    """
    if 'shift' not in params or 'window' not in params:
        return None

    # Reference pulse parameters
    refshape_list = []
    for igain in range(0, get_num_channel(telid)):
        refshape_list.append(get_ref_shapes(telid, igain))
    refshape = np.asarray(refshape_list)
    refstep = get_ref_step(telid)
    nrefstep = get_lrefshape(telid)
    x = np.arange(0, refstep*nrefstep, refstep)
    y = refshape[get_num_channel(telid)-1]
    refipeak = np.argmax(y)
    refpeak = x[refipeak]

    # Sampling pulse parameters
    time_slice = get_time_slice(telid)
    ratio = time_slice/refstep
    x1 = np.arange(0, refstep*nrefstep, time_slice)
    y1 = interp(x1, x, y)
    ipeak = np.argmin(np.abs(x1-x[refipeak]))

    # Check window is within readout
    start = ipeak - params['shift']
    window = params['window']
    if start < 0:
        start = 0
    if start + window > get_num_samples(telid):
        start = get_num_samples(telid) - window

    correction = round((sum(y)*refstep)/(sum(y1[start:start+window])*time_slice),7)
    return correction


def pixel_integration_mc(event, cam, ped, telid, parameters):
    """
    Parameters
    ----------
    event  Data set container to the hess_io event ()
    ped    Array of double containing the pedestal
    telid  Telescope_id
    parameters
    integrator: pixel integration algorithm
       -"full_integration": full digitized range integrated amplitude-pedestal
       -"simple_integration": fixed integration region (window)
       -"global_peak_integration": integration region by global
       peak of significant pixels
       -"local_peak_integration": peak in each pixel determined independently
       -"nb_peak_integration":

    Returns
    -------
    Array of pixels with integrated change [ADC cts], pedestal substracted.
    Returns None if event is None
    """
    if __debug__:
        logger.debug("> %s" % (parameters['integrator']))
    if event is None:
        return None

    switch = {
        'full_integration': lambda: full_integration_mc(event, ped, telid),
        'simple_integration': lambda: simple_integration_mc(
            event, ped, telid, parameters),
        'global_peak_integration': lambda: global_peak_integration_mc(
            event, ped, telid, parameters),
        'local_peak_integration': lambda: local_peak_integration_mc(
            event, ped, telid, parameters),
        'nb_peak_integration': lambda: nb_peak_integration_mc(
            event, cam, ped, telid, parameters),
        }
    try:
        result = switch[parameters['integrator']]()
    except KeyError:
        result = switch[None]()

    return result


def full_integration_mc(event, ped, telid):

    """
    Use full digitized range for the integration amplitude
    algorithm (sum - pedestal)

    No weighting of individual samples is applied.

    Parameters
    ----------

    event  Data set container to the hess_io event ()
    ped    Array of double containing the pedestal
    telid  Telescope_id

    Returns
    -------
    array of pixels with integrated change [ADC cts], pedestal
    substracted per gain

    """

    if event is None or telid < 0:
        return None

    samples_pix_tel_list = []
    for igain in range(0, get_num_channel(telid)):
        samples_pix_tel_list.append(get_adc_sample(telid, igain))
    samples_pix_tel = np.asarray(samples_pix_tel_list, np.int16)
    sum_pix_tel = np.asarray(samples_pix_tel.sum(2)-ped, dtype=np.int16)

    return sum_pix_tel, None


def simple_integration_mc(event, ped, telid, parameters):
    """
    Integrate sample-mode data (traces) over a common and fixed interval.

    The integration window can be anywhere in the available length of
    the traces.
    Since the calibration function subtracts a pedestal that corresponds to the
    total length of the traces we may also have to add a pedestal contribution
    for the samples not summed up.
    No weighting of individual samples is applied.

    Parameters
    ----------

    event  Data set container to the hess_io event ()
    ped    Array of double containing the pedestal
    telid  Telescope_id
    parameters['window']   Number of samples to sum up (is reduced if
                         exceeding available length).
    parameters['shift']  Number of initial samples skipped (adapted such that
                         interval fits into what is available).
    Note: for multiple gains, this results in identical integration regions.

    Returns
    -------
    array of pixels with integrated change [ADC cts], pedestal
    substracted per gain
    """

    if event is None or telid < 0:
        return None
    window = parameters['window']
    shift = parameters['shift']

    # Sanity check on the 'window' and 'shift' parameters given by the "user"
    if (window + shift) > get_num_samples(telid):
        # the number of sample to sum up can not be larger than the actual
        # number of samples of the pixel.
        # If so, the number to sum up is the actual number of samples.
        # the number of samples to skip is calculated again depending on
        # the actual number of samples of the pixel
        if window >= get_num_samples(telid):
            window = get_num_samples(telid)
            shift = 0
        else:
            shift = get_num_samples(telid)-window

    int_corr = np.ones((get_num_channel(telid),
                        get_num_pixels(telid)), dtype=np.int16)
    samples_pix_tel_list = []
    for igain in range(0, get_num_channel(telid)):
        samples_pix_tel_list.append(get_adc_sample(telid, igain))
    samples_pix_tel = np.asarray(samples_pix_tel_list, np.int16)
    samples_pix_win = samples_pix_tel[:, :, shift:window+shift]
    ped_pix_win = ped/get_num_samples(telid)
    sum_pix_tel = np.asarray(int_corr*(samples_pix_win.sum(2) -
                                       ped_pix_win*window), dtype=np.int16)

    return sum_pix_tel, None


def global_peak_integration_mc(event, ped, telid, parameters):
    """
    Integrate sample-mode data (traces) over a common interval around a
    global signal peak.

    The integration window can be anywhere in the available length of the
    traces.
    No weighting of individual samples is applied.

    Parameters
    ----------

    event  Data set container to the hess_io event ()
    ped    Array of double containing the pedestal
    telid  Telescope_id
    parameters['window']    Number of samples to sum up (is reduced if
    exceeding available length).
    parameters['shift'] Start the integration a number of samples before
    the peak, as long as it fits into the available data range.
    Note: for multiple gains, this results in identical integration regions.
    parameters['sigamp']  Amplitude in ADC counts above pedestal at which a
    signal is considered as significant (separate for high gain/low gain).

    Returns
    -------
    array of pixels with integrated change [ADC cts], pedestal
    substracted per gain and peak slide
    """

    # The number of samples to sum up can not be larger than the
    # number of samples
    window = parameters['window']
    if window >= get_num_samples(telid):
        window = get_num_samples(telid)

    samples_pix_tel_list = []
    sigamp_cut = np.ones((get_num_channel(telid)))
    significant_pix = np.ones((get_num_channel(telid),
                               get_num_pixels(telid)), dtype=np.int8)
    for igain in range(0, get_num_channel(telid)):
        samples_pix_tel_list.append(get_adc_sample(telid, igain))
        sigamp_cut[igain] = parameters['sigamp'][igain]
    samples_pix_tel = np.asarray(samples_pix_tel_list, np.int16)
    ped_per_trace = ped/get_num_samples(telid)
    samples_pix_clean = (samples_pix_tel -
                         np.atleast_3d(ped_per_trace)).astype(np.int16)

    # Find the peak (peakpos)
    sigamp_mask = (samples_pix_clean[:] > sigamp_cut)
    # Sample with amplitude larger than 'sigamp' (smaller set to '0')
    samples_pix_filtered = samples_pix_clean*sigamp_mask

    time_pix_tel = samples_pix_filtered.argmax(axis=2)
    max_sample_tel = samples_pix_filtered.max(axis=2)
    significant_pix = significant_pix*(np.any(sigamp_mask, axis=2) == True)

    peakpos = np.zeros((get_num_channel(telid)))
    if np.count_nonzero(significant_pix) > 0 and time_pix_tel.sum(1) > 0:
        peakpos = ((time_pix_tel*max_sample_tel).sum(1) /
                   max_sample_tel.sum(axis=1)).astype(np.int8)
    # Sanitity check
    start = peakpos - parameters['shift']
    if start < 0:
        start = 0
    if start + window > get_num_samples(telid):
        start = get_num_samples(telid) - window

    int_corr = set_integration_correction(telid, parameters)
    # Extract the pulse (pedestal substracted) in the found window
    samples_pix_win = samples_pix_clean[:, :, start:window+start]
    sum_pix_tel = np.asarray(int_corr*(samples_pix_win.sum(2)), dtype=np.int16)

    return sum_pix_tel, time_pix_tel[0]


def local_peak_integration_mc(event, ped, telid, parameters):
    """
    Integrate sample-mode data (traces) around a pixel-local signal peak.

    The integration window can be anywhere in the available
    length of the traces.
    No weighting of individual samples is applied.

    Parameters
    ----------

    event  Data set container to the hess_io event ()
    ped    Array of double containing the pedestal
    telid  Telescope_id
    parameters['window']    Number of samples to sum up (is reduced if
                          exceeding available length).
    parameters['shift'] Start the integration a number of samples before
                        the peak, as long as it fits into the available
                        data range.
    Note: for multiple gains, this results in identical integration regions.
    parameters['sigamp']  Amplitude in ADC counts above pedestal at which a
                          signal is considered as significant (separate for
                          high gain/low gain).

    Returns
    -------
    array of pixels with integrated change [ADC cts], pedestal
    substracted per gain and peak slide
    """

    # The number of samples to sum up can not be larger than the
    # number of samples
    window = parameters['window']
    if window >= get_num_samples(telid):
        window = get_num_samples(telid)

    samples_pix_tel_list = []
    sigamp_cut = np.ones((get_num_channel(telid)))
    significant_pix = np.ones((get_num_channel(telid),
                               get_num_pixels(telid)), dtype=np.int8)
    for igain in range(0, get_num_channel(telid)):
        samples_pix_tel_list.append(get_adc_sample(telid, igain))
        sigamp_cut[igain] = parameters['sigamp'][igain]
    samples_pix_tel = np.asarray(samples_pix_tel_list, np.int16)
    ped_per_trace = ped/get_num_samples(telid)
    samples_pix_clean = (samples_pix_tel -
                         np.atleast_3d(ped_per_trace)).astype(np.int16)

    # Find the peak (peakpos)
    sigamp_mask = (samples_pix_clean[:] > sigamp_cut)
    # Sample with amplitude larger than 'sigamp'
    samples_pix_filtered = samples_pix_clean*sigamp_mask
    time_pix_tel = samples_pix_filtered.argmax(axis=2)
    max_sample_tel = samples_pix_filtered.max(axis=2)
    significant_pix = significant_pix*(np.any(sigamp_mask, axis=2) == True)

    # If the LG is not significant, takes the HG peakpos
    peakpos = time_pix_tel*significant_pix
    if get_num_channel(telid) > 1:
        peakpos[0] = np.where(significant_pix[0] < significant_pix[1],
                              time_pix_tel[1], time_pix_tel[0]).astype(np.int8)

    # Sanitity check
    start = peakpos - parameters['shift']
    start[start < 0] = 0
    start[start + window > get_num_samples(telid)] = get_num_samples(telid)
    - window

    int_corr = set_integration_correction(telid, parameters)
    # Create a mask with the integration windows per pixel
    m = np.zeros_like(samples_pix_clean)
    for i in range(0, np.shape(samples_pix_clean)[0]):
        for j in range(0, np.shape(samples_pix_clean)[1]):
            m[i, j, start[i, j]:start[i, j]+window] = 1
    samples_pix_win = samples_pix_clean*m
    # Extract the pulse (pedestal substracted) in the found window
    sum_pix_tel = np.asarray(int_corr*(samples_pix_win.sum(2)), dtype=np.int16)

    return sum_pix_tel, time_pix_tel[0]


def nb_peak_integration_mc(event, cam, ped, telid, parameters):

    """
    Integrate sample-mode data (traces) around a peak in the signal sum of
    neighbouring pixels.

    The integration window can be anywhere in the available length
    of the traces.
    No weighting of individual samples is applied.

    Parameters
    ----------

    event                 Data set container to the hess_io event ()
    cam                   Data set container with the camera information
    ped                   Array of double containing the pedestal
    telid                 Telescope_id
    parameters['window']    Number of samples to sum up
                          (is reduced if exceeding available length).
    parameters['shift'] Start the integration a number of samples before
                          the peak, as long as it fits into the available data
                          range.
                          Note: for multiple gains, this results in identical
                          integration regions.
    parameters['sigamp']  Amplitude in ADC counts above pedestal at which
                          a signal is considered as significant (separate for
                          high gain/low gain).
    parameters['lwt']     Weight of the local pixel (0: peak from neighbours
                          only,1: local pixel counts as much as any neighbour).

    Returns
    -------
    array of pixels with integrated change [ADC cts], pedestal
    substracted per gain and peak slide
    """

    istart = time.process_time()
    # The number of samples to sum up can not be larger than
    # the number of samples
    window = parameters['window']
    if window >= get_num_samples(telid):
        window = get_num_samples(telid)
    lwt = parameters['lwt']

    #  For this integration scheme we need the list of neighbours early on
    pix_x, pix_y = event.meta.pixel_pos[telid]
    foclen = event.meta.optical_foclen[telid]
    geom = io.CameraGeometry.guess(pix_x, pix_y, foclen)
    # geom = cam['CameraTable_VersionFeb2016_TelID%s'%telid]

    iend = time.process_time()

    samples_pix_tel_list = []
    sigamp_cut = np.ones((get_num_channel(telid)))
    time_pix_tel = np.ones((get_num_channel(telid),
                            get_num_pixels(telid)), dtype=np.int8)
    for igain in range(0, get_num_channel(telid)):
        samples_pix_tel_list.append(get_adc_sample(telid, igain))
        sigamp_cut[igain] = parameters['sigamp'][igain]
    samples_pix_tel = np.asarray(samples_pix_tel_list, np.int16)
    ped_per_trace = ped/get_num_samples(telid)
    samples_pix_clean = (samples_pix_tel-np.atleast_3d(ped_per_trace)
                         ).astype(np.int16)

    int_corr = set_integration_correction(telid, parameters)
    # Create a mask with the integration windows per pixel and per gain
    m = np.zeros_like(samples_pix_clean)
    for i in range(0, np.shape(samples_pix_clean)[0]):
        for j in range(0, np.shape(samples_pix_clean)[1]):
            nb_samples = samples_pix_clean[i, geom.neighbors[j], :]
            # nb_samples = samples_pix_clean[i, geom['PixNeig'][j], :]
            all_samples = np.vstack([nb_samples,
                                     lwt*samples_pix_clean[i, j, :]])
            sum_samples = all_samples.sum(0)
            peakpos = np.mean(sum_samples.argmax(0)).astype(np.int8)
            time_pix_tel[i, j] = peakpos
            start = peakpos - parameters['shift']
            m[i, j, start:start+window] = 1

    samples_pix_win = samples_pix_clean*m
    # Extract the pulse (pedestal substracted) in the found window
    sum_pix_tel = np.asarray(int_corr*(samples_pix_win.sum(2)), dtype=np.int16)

    if __debug__: print(" inter-integration %.3e sec"%(iend-istart))

    print(sum_pix_tel[0][0],time_pix_tel[0][0])
    return sum_pix_tel, time_pix_tel[0]


def calibrate_amplitude_mc(integrated_charge, calib, telid, params):
    """
    Parameters
    ----------
    integrated_charge     Array of pixels with integrated change [ADC cts],
                          pedestal substracted
    calib                 Array of double containing the single-pe events
    parameters['clip_amp']  Amplitude in p.e. above which the signal is
                            clipped.
    Returns
    ------
    Array of pixels with calibrate charge [photo-electrons]
    Returns None if event is None

    """

    if integrated_charge is None:
        return None

    pe_pix_tel = []
    for ipix in range(0, get_num_pixels(telid)):
        pe_pix = 0
        int_pix_hg = integrated_charge[get_num_channel(telid)-1][ipix]
        # If the integral charge is between -300,2000 ADC cts, we choose the HG
        # Otherwise the LG channel
        # If there is only one gain, it is the HG (default)
        if ((int_pix_hg > -1000) and
            (int_pix_hg < 10000) or
            (get_num_channel(telid) < 2)):
            pe_pix = (integrated_charge[get_num_channel(telid)-1][ipix] *
                      calib[get_num_channel(telid)-1][ipix])
        else:
            pe_pix = (integrated_charge[get_num_channel(telid)][ipix] *
                      calib[get_num_channel(telid)][ipix])

        if "climp_amp" in params and params["clip_amp"] > 0:
            if pe_pix > params["clip_amp"]:
                pe_pix = params["clip_amp"]

        # pe_pix is in units of 'mean photo-electrons'
        # (unit = mean p.e. signal.).
        # We convert to experimentalist's 'peak photo-electrons'
        # now (unit = most probable p.e. signal after experimental resolution).
        # Keep in mind: peak(10 p.e.) != 10*peak(1 p.e.)
        pe_pix_tel.append(pe_pix*CALIB_SCALE)

    return np.asarray(pe_pix_tel)