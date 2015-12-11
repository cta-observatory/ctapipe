"""
Integrate sample-mode data (traces) Functions
and
Convert the integral pixel ADC count to photo-electrons
"""

import sys
import numpy as np
from pyhessio import *
from ctapipe import io
from astropy import units as u

__all__ = [
    'set_integration_correction',
    'pixel_integration_mc',
    'full_integration_mc',
    'simple_integration_mc',
    'global_peak_integration_mc',
    'local_peak_integration_mc',
    'nb_peak_integration_mc',
    'calibrate_amplitude_mc',
]

CALIB_SCALE = 0.92

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


def qpol(x, np, yval):
    ix = round(x)
    if x < 0 or x >= float(np):
        return 0.
    if ix+1 >= np:
        return 0.
    return yval[ix]*(ix+1-x) + yval[ix-1]*(x-ix)


def set_integration_correction(telid, params):
    TAG = sys._getframe().f_code.co_name+">"
    """
    Parameters
    ----------
    event  Data set container to the hess_io event ()
    telid  Telescope_id
    nbins
    parameters['nskip']  Number of initial samples skipped
    (adapted such that interval fits into what is available).
    Returns
    -------
    Array of gains with the integration correction [ADC cts]
    Returns None if parameters do not include 'nskip'
    """
    if 'nskip' not in params or 'nsum' not in params:
        return None

    integration_correction = []
    for igain in range(0, get_num_channel(telid)):
        refshape = get_ref_shapes(telid, igain)
        int_corr = 1.0
        # Sum over all the pulse we have and rescale to original time step
        asum = sum(refshape)*get_ref_step(telid)/get_time_slice(telid)
        # Find the pulse shape peak (bin and value)
        speak = max(refshape)
        ipeak = refshape.argmax(axis=0)
        # Sum up given interval starting from peak, averaging over phase
        around_peak_sum = 0
        for iphase in range(0, 5):
            ti = (((iphase*0.2-0.4) - params['nskip']) *
            get_time_slice(telid)/get_ref_step(telid) + ipeak)
            for ibin in range(0, params['nsum']):
                around_peak_sum += qpol(
                    ibin*get_time_slice(telid)/get_ref_step(telid) +
                    ti, get_lrefshape(telid), refshape)
        around_peak_sum *= 0.2
        if around_peak_sum > 0. and asum > 0.:
            int_corr = asum/around_peak_sum

        integration_correction.append(int_corr)

    return integration_correction


def pixel_integration_mc(event, ped, telid, parameters):
    TAG = sys._getframe().f_code.co_name+">"
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
        print(TAG, parameters['integrator'], end="\n")
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
            event, ped, telid, parameters),
        }
    try:
        result = switch[parameters['integrator']]()
    except KeyError:
        result = switch[None]()

    return result


def full_integration_mc(event, ped, telid):
    TAG = sys._getframe().f_code.co_name+">"

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

    sum_pix_tel = []
    for igain in range(0, get_num_channel(telid)):
        sum_pix = []
        for ipix in range(0, get_num_pixels(telid)):
            samples_pix = get_adc_sample(telid, igain)[ipix]
            sum_pix.append(sum(samples_pix[:])-ped[igain][ipix])
        sum_pix_tel.append(sum_pix)

    return sum_pix_tel, None


def simple_integration_mc(event, ped, telid, parameters):
    TAG = sys._getframe().f_code.co_name+">"
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
    parameters['nsum']   Number of samples to sum up (is reduced if
                         exceeding available length).
    parameters['nskip']  Number of initial samples skipped (adapted such that
                         interval fits into what is available).
    Note: for multiple gains, this results in identical integration regions.

    Returns
    -------
    array of pixels with integrated change [ADC cts], pedestal
    substracted per gain
    """

    if event is None or telid < 0:
        return None
    nsum = parameters['nsum']
    nskip = parameters['nskip']

    # Sanity check on the 'nsum' and 'nskip' parameters given by the "user"
    if (nsum + nskip) > get_num_samples(telid):
        # the number of sample to sum up can not be larger than the actual
        # number of samples of the pixel.
        # If so, the number to sum up is the actual number of samples.
        # the number of samples to skip is calculated again depending on
        # the actual number of samples of the pixel
        if nsum >= get_num_samples(telid):
            nsum = get_num_samples(telid)
            nskip = 0
        else:
            nskip = get_num_samples(telid)-nsum

    sum_pix_tel = []
    int_corr = 1.
    for igain in range(0, get_num_channel(telid)):
        sum_pix = []
        for ipix in range(0, get_num_pixels(telid)):
            samples_pix_win = (get_adc_sample(telid, igain)[ipix]
            [nskip:(nsum+nskip)])
            ped_per_trace = ped[igain][ipix]/get_num_samples(telid)
            sum_pix.append(int(int_corr[igain]*(sum(samples_pix_win) -
                                                ped_per_trace*nsum)+0.5))
        sum_pix_tel.append(sum_pix)

    return sum_pix_tel, None


def global_peak_integration_mc(event, ped, telid, parameters):
    TAG = sys._getframe().f_code.co_name+">"
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
    parameters['nsum']    Number of samples to sum up (is reduced if
    exceeding available length).
    parameters['nskip'] Start the integration a number of samples before
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
    nsum = parameters['nsum']
    if nsum >= get_num_samples(telid):
        nsum = get_num_samples(telid)

    sum_pix_tel = []
    time_pix_tel = []
    jpeak = []
    ppeak = []
    for igain in range(0, get_num_channel(telid)):
        peakpos = 0
        npeaks = 0
        # Find the peak (peakpos)
        peak_pix = []
        for ipix in range(0, get_num_pixels(telid)):
            ped_per_trace = int(ped[igain][ipix]/get_num_samples(telid)+0.5)
            samples_pix_clean = get_adc_sample(telid, igain)
            [ipix].astype(np.int16) - ped_per_trace
            significant = 0
            ipeak = -1
            for isamp in range(0, get_num_samples(telid)):
                if samples_pix_clean[isamp] >= parameters['sigamp'][igain]:
                    significant = 1
                    ipeak = isamp
                    for isamp2 in range(isamp+1, get_num_samples(telid)):
                        if (samples_pix_clean[isamp2] >
                        samples_pix_clean[ipeak]):
                            ipeak = isamp2
                    break

            if significant == 1:
                jpeak.append(ipeak)
                ppeak.append(samples_pix_clean[ipeak])
                npeaks += 1
                peak_pix.append(ipeak)

        peakpos = 0
        if npeaks > 0 and sum(jpeak) > 0.:
            peakpos = sum(np.array(jpeak)*np.array(ppeak))/sum(np.array(ppeak))

        # Sanitity check
        start = round(peakpos) - parameters['nskip']
        if start < 0:
            start = 0
        if start + nsum > get_num_samples(telid):
            start = get_num_samples(telid) - nsum

        int_corr = set_integration_correction(telid, parameters)
        # Integrate pixel
        sum_pix = []
        for ipix in range(0, get_num_pixels(telid)):
            samples_pix_win = get_adc_sample(telid, igain)
            [ipix][start:(nsum+start)]
            ped_per_trace = ped[igain][ipix]/get_num_samples(telid)
            sum_pix.append(round(int_corr[igain] *
                                 (sum(samples_pix_win) - ped_per_trace*nsum)))

        sum_pix_tel.append(sum_pix)
        time_pix_tel.append(peak_pix)
    return sum_pix_tel, time_pix_tel


def local_peak_integration_mc(event, ped, telid, parameters):
    TAG = sys._getframe().f_code.co_name+">"
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
    parameters['nsum']    Number of samples to sum up (is reduced if
                          exceeding available length).
    parameters['nskip'] Start the integration a number of samples before
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

    # The number of samples to sum up can not be larger than
    # the number of samples
    nsum = parameters['nsum']
    if nsum >= get_num_samples(telid):
        nsum = get_num_samples(telid)

    sum_pix_tel = []
    time_pix_tel = []
    peakpos = []  # [igain][ipix] (def. io_hess.h: 0=HI_GAIN,1=LO_GAIN)
    for igain in range(0, get_num_channel(telid)):
        jpeak = []  # [ipix] local peak for each pixel
        sum_pix = []
        peak_pix = []
        for ipix in range(0, get_num_pixels(telid)):
            # Find the peak (peakpos)
            ped_per_trace = int(ped[igain][ipix]/get_num_samples(telid)+0.5)
            samples_pix_clean = get_adc_sample(telid, igain)
            [ipix].astype(np.int16) - ped_per_trace
            significant = 0
            ipeak = -1
            for isamp in range(0, get_num_samples(telid)):
                if samples_pix_clean[isamp] >= parameters['sigamp'][igain]:
                    significant = 1
                    ipeak = isamp
                    for isamp2 in range(isamp+1, get_num_samples(telid)):
                        if (samples_pix_clean[isamp2] >
                        samples_pix_clean[ipeak]):
                            ipeak = isamp2
                    break
            peak_pix.append(ipeak)
            if igain == 0:
                jpeak.append(ipeak)
            else:
                # If the LG is not significant, takes the HG peakpos
                if significant and ipeak >= 0:
                    jpeak.append(ipeak)
                else:
                    jpeak.append(peakpos[0][ipix])

            # Sanitity check
            start = round(jpeak[ipix]) - parameters['nskip']
            if start < 0:
                start = 0
            if start + nsum > get_num_samples(telid):
                start = get_num_samples(telid) - nsum

            int_corr = set_integration_correction(telid, parameters)
            # Integrate pixel
            samples_pix_win = get_adc_sample(telid, igain)
            [ipix][start:(nsum+start)]
            ped_per_trace = ped[igain][ipix]/get_num_samples(telid)
            if jpeak[ipix] > 0:
                sum_pix.append(round(int_corr[igain] *
                                     (sum(samples_pix_win) -
                                      ped_per_trace*nsum)))
            else:
                sum_pix.append(0.)

        # Save the peak positions for LG check
        peakpos.append(jpeak)

        sum_pix_tel.append(sum_pix)
        time_pix_tel.append(peak_pix)
    return sum_pix_tel, time_pix_tel


def nb_peak_integration_mc(event, ped, telid, parameters):
    TAG = sys._getframe().f_code.co_name+">"
    """
    Integrate sample-mode data (traces) around a peak in the signal sum of
    neighbouring pixels.

    The integration window can be anywhere in the available length
    of the traces.
    No weighting of individual samples is applied.

    Parameters
    ----------

    event                 Data set container to the hess_io event ()
    ped                   Array of double containing the pedestal
    telid                 Telescope_id
    parameters['nsum']    Number of samples to sum up
                          (is reduced if exceeding available length).
    parameters['nskip'] Start the integration a number of samples before
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

    # The number of samples to sum up can not be larger than
    # the number of samples
    nsum = parameters['nsum']
    if nsum >= get_num_samples(telid):
        nsum = get_num_samples(telid)

    #  For this integration scheme we need the list of neighbours early on
    pix_x, pix_y = event.meta.pixel_pos[telid]
    geom = io.CameraGeometry.guess(pix_x*u.m, pix_y*u.m)

    sum_pix_tel = []
    time_pix_tel = []
    for igain in range(0, get_num_channel(telid)):
        sum_pix = []
        peak_pix = []
        for ipix in range(0, get_num_pixels(telid)):
            i = 0
            knb = 0
            # Loop over the neighbors of ipix
            ipix_nb = geom.neighbors[ipix]
            nb_samples = [0 for ii in range(get_num_samples(telid))]
            for inb in range(len(ipix_nb)):
                nb_samples += np.array(get_adc_sample(telid, igain)
                                       [ipix_nb[inb]])
                knb += 1
            if parameters['lwt'] > 0:
                for isamp in range(1, get_num_samples(telid)):
                    nb_samples += np.array(get_adc_sample(telid, igain)
                                           [ipix])*lwt
                knb += 1

            if knb == 0:
                continue
            ipeak = 0
            p = nb_samples[0]
            for isamp in range(1, get_num_samples(telid)):
                if nb_samples[isamp] > p:
                    p = nb_samples[isamp]
                    ipeak = isamp
            peakpos = peakpos_hg = ipeak
            start = peakpos - parameters['nskip']

            # Sanitity check?
            if start < 0:
                start = 0
            if start + nsum > get_num_samples(telid):
                start = get_num_samples(telid) - nsum

            int_corr = set_integration_correction(telid, parameters)
            # Integrate pixel
            samples_pix_win = get_adc_sample(telid, igain)
            [ipix][start:(nsum+start)]
            ped_per_trace = ped[igain][ipix]/get_num_samples(telid)
            sum_pix.append(round(int_corr[igain]*(sum(samples_pix_win) -
                                                  ped_per_trace*nsum)))
            peak_pix.append(peakpos)

        sum_pix_tel.append(sum_pix)
        time_pix_tel.append(peak_pix)

    return sum_pix_tel, time_pix_tel


def calibrate_amplitude_mc(integrated_charge, calib, telid, params):
    TAG = sys._getframe().f_code.co_name+">"
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
        # If the integral charge is between -300,2000 ADC ts, we choose the HG
        # Otherwise the LG channel
        # If there is only one gain, it is the HG (default)
        if (int_pix_hg > -1000 and int_pix_hg < 10000 or
        get_num_channel(telid) < 2):
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

    return pe_pix_tel
