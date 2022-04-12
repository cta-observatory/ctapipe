"""
Charge extraction algorithms to reduce the image to one value per pixel
"""

__all__ = [
    "ImageExtractor",
    "FullWaveformSum",
    "FixedWindowSum",
    "GlobalPeakWindowSum",
    "LocalPeakWindowSum",
    "SlidingWindowMaxSum",
    "NeighborPeakWindowSum",
    "BaselineSubtractedNeighborPeakWindowSum",
    "TwoPassWindowSum",
    "extract_around_peak",
    "extract_sliding_window",
    "neighbor_average_waveform",
    "subtract_baseline",
    "integration_correction",
]


from abc import abstractmethod
from functools import lru_cache
from typing import Tuple

import numpy as np
from ctapipe.containers import DL1CameraContainer
from ctapipe.core import TelescopeComponent
from ctapipe.core.traits import (
    BoolTelescopeParameter,
    FloatTelescopeParameter,
    IntTelescopeParameter,
)
from numba import float32, float64, guvectorize, int64, njit, prange
from scipy.ndimage import convolve1d
from traitlets import Bool, Int

from . import brightest_island, number_of_islands, tailcuts_clean
from .hillas import camera_to_shower_coordinates, hillas_parameters
from .timing import timing_parameters


@guvectorize(
    [
        (float64[:], int64, int64, int64, float64, float32[:], float32[:]),
        (float32[:], int64, int64, int64, float64, float32[:], float32[:]),
    ],
    "(s),(),(),(),()->(),()",
    nopython=True,
    cache=True,
)
def extract_around_peak(
    waveforms, peak_index, width, shift, sampling_rate_ghz, sum_, peak_time
):
    """
    This function performs the following operations:

    - Sum the samples from the waveform using the window defined by a
    peak position, window width, and window shift.
    - Obtain the pulse time within a window defined by a peak finding
    algorithm, using the weighted average of the samples.

    This function is a numpy universal function which defines the operation
    applied on the waveform for every channel and pixel. Therefore in the
    code body of this function:
        - waveforms is a 1D array of size n_samples.
        - peak_index, width and shift are integers, corresponding to the
            correct value for the current pixel

    The ret argument is required by numpy to create the numpy array which is
    returned. It can be ignored when calling this function.

    Parameters
    ----------
    waveforms : ndarray
        Waveforms stored in a numpy array.
        Shape: (n_pix, n_samples)
    peak_index : ndarray or int
        Peak index for each pixel.
    width : ndarray or int
        Window size of integration window for each pixel.
    shift : ndarray or int
        Window size of integration window for each pixel.
    sampling_rate_ghz : float
        Sampling rate of the camera, in units of GHz
        Astropy units should have to_value('GHz') applied before being passed
    sum_ : ndarray
        Return argument for ufunc (ignore)
        Returns the sum of the waveform samples
    peak_time : ndarray
        Return argument for ufunc (ignore)
        Returns the peak_time in units "ns"

    Returns
    -------
    charge : ndarray
        Extracted charge.
        Shape: (n_pix)

    """
    n_samples = waveforms.size
    start = peak_index - shift
    end = start + width

    # reduce to valid range
    start = max(0, start)
    end = min(end, n_samples)

    i_sum = float64(0.0)
    time_num = float64(0.0)
    time_den = float64(0.0)

    for isample in prange(start, end):
        i_sum += waveforms[isample]
        if waveforms[isample] > 0:
            time_num += waveforms[isample] * isample
            time_den += waveforms[isample]

    peak_time[0] = time_num / time_den if time_den > 0 else peak_index
    # Convert to units of ns
    peak_time[0] /= sampling_rate_ghz
    sum_[0] = i_sum


@guvectorize(
    [
        (float64[:], int64, float64, float32[:], float32[:]),
        (float32[:], int64, float64, float32[:], float32[:]),
    ],
    "(s),(),()->(),()",
    nopython=True,
)
def extract_sliding_window(waveforms, width, sampling_rate_ghz, sum_, peak_time):
    """
    This function performs the following operations:

    - Find the largest sum of width consecutive slices
    - Obtain the pulse time within a window defined by a peak finding
    algorithm, using the weighted average of the samples.

    This function is a numpy universal function which defines the operation
    applied on the waveform for every channel and pixel. Therefore in the
    code body of this function:
        - waveforms is a 1D array of size n_samples.
        - width is integer

    The ret argument is required by numpy to create the numpy array which is
    returned. It can be ignored when calling this function.

    Parameters
    ----------
    waveforms : ndarray
        Waveforms stored in a numpy array.
        Shape: (n_pix, n_samples)
    width : ndarray or int
        Window size of integration window for each pixel.
    sampling_rate_ghz : float
        Sampling rate of the camera, in units of GHz
        Astropy units should have to_value('GHz') applied before being passed
    sum_ : ndarray
        Return argument for ufunc (ignore)
        Returns the sum of the waveform samples
    peak_time : ndarray
        Return argument for ufunc (ignore)
        Returns the peak_time in units "ns"

    Returns
    -------
    charge : ndarray
        Extracted charge.
        Shape: (n_pix)

    """

    # first find the cumulative waveform
    cwf = np.cumsum(waveforms)
    # add zero at the begining so it is easier to substract the two arrays later
    cwf = np.concatenate((np.zeros(1), cwf))
    sums = cwf[width:] - cwf[:-width]
    maxpos = np.argmax(sums)  # start of the window with largest sum
    sum_[0] = sums[maxpos]

    time_num = float64(0.0)
    time_den = float64(0.0)
    # now compute the timing as the average of non negative slices
    for isample in prange(maxpos, maxpos + width):
        if waveforms[isample] > 0:
            time_num += waveforms[isample] * isample
            time_den += waveforms[isample]

    peak_time[0] = time_num / time_den if time_den > 0 else maxpos + 0.5 * width
    # Convert to units of ns
    peak_time[0] /= sampling_rate_ghz


@njit(parallel=True, cache=True)
def neighbor_average_waveform(waveforms, neighbors_indices, neighbors_indptr, lwt):
    """
    Obtain the average waveform built from the neighbors of each pixel

    Parameters
    ----------
    waveforms : ndarray
        Waveforms stored in a numpy array.
        Shape: (n_pix, n_samples)
    neighbors_indices : ndarray
        indices of a scipy csr sparse matrix of neighbors, i.e.
        ``ctapipe.instrument.CameraGeometry.neighbor_matrix_sparse.indices``.
    neighbors_indptr : ndarray
        indptr of a scipy csr sparse matrix of neighbors, i.e.
        ``ctapipe.instrument.CameraGeometry.neighbor_matrix_sparse.indptr``.
    lwt: int
        Weight of the local pixel (0: peak from neighbors only,
        1: local pixel counts as much as any neighbor)

    Returns
    -------
    average_wf : ndarray
        Average of neighbor waveforms for each pixel.
        Shape: (n_pix, n_samples)

    """

    n_pixels = waveforms.shape[0]
    indptr = neighbors_indptr
    indices = neighbors_indices

    # initialize to waveforms weighted with lwt
    # so the value of the pixel itself is already taken into account
    average = waveforms * lwt

    for pixel in prange(n_pixels):
        neighbors = indices[indptr[pixel] : indptr[pixel + 1]]

        n = lwt
        for neighbor in neighbors:
            average[pixel] += waveforms[neighbor]
            n += 1

        average[pixel] /= n

    return average


def subtract_baseline(waveforms, baseline_start, baseline_end):
    """
    Subtracts the waveform baseline, estimated as the mean waveform value
    in the interval [baseline_start:baseline_end]

    Parameters
    ----------
    waveforms : ndarray
        Waveforms stored in a numpy array.
        Shape: (n_pix, n_samples)
    baseline_start : int
        Sample where the baseline window starts
    baseline_end : int
        Sample where the baseline window ends

    Returns
    -------
    baseline_corrected : ndarray
        Waveform with the baseline subtracted
    """
    baseline_corrected = (
        waveforms
        - np.mean(waveforms[..., baseline_start:baseline_end], axis=-1)[..., None]
    )

    return baseline_corrected


def integration_correction(
    reference_pulse_shape,
    reference_pulse_sample_width_ns,
    sample_width_ns,
    window_width,
    window_shift,
):
    """
    Obtain the correction for the integration window specified.

    For any integration window applied to a noise-less unit pulse, the
    correction (returned by this function) multiplied by the integration
    result should equal 1.

    This correction therefore corrects for the Cherenkov signal that may be
    outside the integration window, and removes any dependence of the resulting
    image on the window_width and window_shift parameters. However, the width
    and shift of the window should still be optimised for the pulse finding and
    to minimise the noise included in the integration.

    Parameters
    ----------
    reference_pulse_shape : ndarray
        Numpy array containing the pulse shape for each gain channel
    reference_pulse_sample_width_ns : float
        The width of the reference pulse sample time bin in ns
    sample_width_ns : float
        The width of the waveform sample time bin in ns
    window_width : int
        Width of the integration window (in units of n_samples)
    window_shift : int
        Shift to before the peak for the start of the integration window
        (in units of n_samples)

    Returns
    -------
    correction : ndarray
        Value of the integration correction for each gain channel
    """
    n_channels = len(reference_pulse_shape)
    correction = np.ones(n_channels, dtype=np.float64)
    for ichannel, pulse_shape in enumerate(reference_pulse_shape):
        pulse_max_sample = pulse_shape.size * reference_pulse_sample_width_ns
        pulse_shape_x = np.arange(0, pulse_max_sample, reference_pulse_sample_width_ns)
        sampled_edges = np.arange(0, pulse_max_sample, sample_width_ns)

        sampled_pulse, _ = np.histogram(
            pulse_shape_x, sampled_edges, weights=pulse_shape, density=True
        )
        n_samples = sampled_pulse.size
        start = sampled_pulse.argmax() - window_shift
        start = start if start >= 0 else 0
        end = start + window_width
        end = end if end < n_samples else n_samples
        if start >= end:
            continue

        integration = sampled_pulse[start:end] * sample_width_ns
        correction[ichannel] = 1.0 / np.sum(integration)

    return correction


class ImageExtractor(TelescopeComponent):
    def __init__(self, subarray, config=None, parent=None, **kwargs):
        """
        Base component to handle the extraction of charge and pulse time
        from an image cube (waveforms), taking into account the sampling rate
        of the waveform.

        Assuming a waveform with sample units X and containing a noise-less unit
        pulse, the aim of the ImageExtractor is to return 1 X*ns.

        Parameters
        ----------
        subarray: ctapipe.instrument.SubarrayDescription
            Description of the subarray. Provides information about the
            camera which are useful in charge extraction, such as reference
            pulse shape, sampling rate, neighboring pixels. Also required for
            configuring the TelescopeParameter traitlets.
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool : ctapipe.core.Tool or None
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        kwargs
        """
        super().__init__(subarray=subarray, config=config, parent=parent, **kwargs)

        self.sampling_rate_ghz = {
            telid: telescope.camera.readout.sampling_rate.to_value("GHz")
            for telid, telescope in subarray.tel.items()
        }

    @abstractmethod
    def __call__(self, waveforms, telid, selected_gain_channel) -> DL1CameraContainer:
        """
        Call the relevant functions to fully extract the charge and time
        for the particular extractor.

        Parameters
        ----------
        waveforms : ndarray
            Waveforms stored in a numpy array of shape
            (n_pix, n_samples).
        telid : int
            The telescope id. Used to obtain to correct traitlet configuration
            and instrument properties
        selected_gain_channel : ndarray
            The channel selected in the gain selection, per pixel. Required in
            some cases to calculate the correct correction for the charge
            extraction.

        Returns
        -------
        DL1CameraContainer:
            extracted images and validity flags
        """


class FullWaveformSum(ImageExtractor):
    """
    Extractor that sums the entire waveform.
    """

    def __call__(self, waveforms, telid, selected_gain_channel):
        charge, peak_time = extract_around_peak(
            waveforms, 0, waveforms.shape[-1], 0, self.sampling_rate_ghz[telid]
        )
        return DL1CameraContainer(image=charge, peak_time=peak_time, is_valid=True)


class FixedWindowSum(ImageExtractor):
    """
    Extractor that sums within a fixed window defined by the user.
    """

    peak_index = IntTelescopeParameter(
        default_value=0, help="Manually select index where the peak is located"
    ).tag(config=True)
    window_width = IntTelescopeParameter(
        default_value=7, help="Define the width of the integration window"
    ).tag(config=True)
    window_shift = IntTelescopeParameter(
        default_value=0,
        help="Define the shift of the integration window from the peak_index "
        "(peak_index - shift)",
    ).tag(config=True)

    apply_integration_correction = BoolTelescopeParameter(
        default_value=True, help="Apply the integration window correction"
    ).tag(config=True)

    @lru_cache(maxsize=128)
    def _calculate_correction(self, telid):
        """
        Calculate the correction for the extracted change such that the value
        returned would equal 1 for a noise-less unit pulse.

        This method is decorated with @lru_cache to ensure it is only
        calculated once per telescope.

        Parameters
        ----------
        telid : int

        Returns
        -------
        correction : ndarray
        The correction to apply to an extracted charge using this ImageExtractor
        Has size n_channels, as a different correction value might be required
        for different gain channels.
        """
        readout = self.subarray.tel[telid].camera.readout
        return integration_correction(
            readout.reference_pulse_shape,
            readout.reference_pulse_sample_width.to_value("ns"),
            (1 / readout.sampling_rate).to_value("ns"),
            self.window_width.tel[telid],
            self.window_shift.tel[telid],
        )

    def __call__(self, waveforms, telid, selected_gain_channel):
        charge, peak_time = extract_around_peak(
            waveforms,
            self.peak_index.tel[telid],
            self.window_width.tel[telid],
            self.window_shift.tel[telid],
            self.sampling_rate_ghz[telid],
        )
        if self.apply_integration_correction.tel[telid]:
            charge *= self._calculate_correction(telid=telid)[selected_gain_channel]
        return DL1CameraContainer(image=charge, peak_time=peak_time, is_valid=True)


class GlobalPeakWindowSum(ImageExtractor):
    """
    Extractor which sums in a window about the
    peak from the global average waveform.

    To reduce the influence of noise pixels, the average can be calculated
    only on the ``pixel_fraction`` brightest pixels.
    The "brightest" pixels are determined by sorting the waveforms by their
    maximum value.
    """

    window_width = IntTelescopeParameter(
        default_value=7, help="Define the width of the integration window"
    ).tag(config=True)

    window_shift = IntTelescopeParameter(
        default_value=3,
        help="Define the shift of the integration window from the peak_index "
        "(peak_index - shift)",
    ).tag(config=True)

    apply_integration_correction = BoolTelescopeParameter(
        default_value=True, help="Apply the integration window correction"
    ).tag(config=True)

    pixel_fraction = FloatTelescopeParameter(
        default_value=1.0,
        help=(
            "Fraction of pixels to use for finding the integration window."
            " By default, the full camera is used."
            " If fraction is smaller 1, only the brightest pixels will be averaged"
            " to find the peak position"
        ),
    ).tag(config=True)

    @lru_cache(maxsize=128)
    def _calculate_correction(self, telid):
        """
        Calculate the correction for the extracted change such that the value
        returned would equal 1 for a noise-less unit pulse.

        This method is decorated with @lru_cache to ensure it is only
        calculated once per telescope.

        Parameters
        ----------
        telid : int

        Returns
        -------
        correction : ndarray
        The correction to apply to an extracted charge using this ImageExtractor
        Has size n_channels, as a different correction value might be required
        for different gain channels.
        """
        readout = self.subarray.tel[telid].camera.readout
        return integration_correction(
            readout.reference_pulse_shape,
            readout.reference_pulse_sample_width.to_value("ns"),
            (1 / readout.sampling_rate).to_value("ns"),
            self.window_width.tel[telid],
            self.window_shift.tel[telid],
        )

    def __call__(self, waveforms, telid, selected_gain_channel):
        if self.pixel_fraction.tel[telid] == 1.0:
            # average over pixels then argmax over samples
            peak_index = waveforms.mean(axis=-2).argmax()
        else:
            n_pixels = int(self.pixel_fraction.tel[telid] * waveforms.shape[-2])
            brightest = np.argsort(waveforms.max(axis=-1))[..., -n_pixels:]

            # average over brightest pixels then argmax over samples
            peak_index = waveforms[brightest].mean(axis=-2).argmax()

        charge, peak_time = extract_around_peak(
            waveforms,
            peak_index,
            self.window_width.tel[telid],
            self.window_shift.tel[telid],
            self.sampling_rate_ghz[telid],
        )
        if self.apply_integration_correction.tel[telid]:
            charge *= self._calculate_correction(telid=telid)[selected_gain_channel]
        return DL1CameraContainer(image=charge, peak_time=peak_time, is_valid=True)


class LocalPeakWindowSum(ImageExtractor):
    """
    Extractor which sums in a window about the
    peak in each pixel's waveform.
    """

    window_width = IntTelescopeParameter(
        default_value=7, help="Define the width of the integration window"
    ).tag(config=True)

    window_shift = IntTelescopeParameter(
        default_value=3,
        help="Define the shift of the integration window"
        "from the peak_index (peak_index - shift)",
    ).tag(config=True)

    apply_integration_correction = BoolTelescopeParameter(
        default_value=True, help="Apply the integration window correction"
    ).tag(config=True)

    @lru_cache(maxsize=128)
    def _calculate_correction(self, telid):
        """
        Calculate the correction for the extracted change such that the value
        returned would equal 1 for a noise-less unit pulse.

        This method is decorated with @lru_cache to ensure it is only
        calculated once per telescope.

        Parameters
        ----------
        telid : int

        Returns
        -------
        correction : ndarray
        The correction to apply to an extracted charge using this ImageExtractor
        Has size n_channels, as a different correction value might be required
        for different gain channels.
        """
        readout = self.subarray.tel[telid].camera.readout
        return integration_correction(
            readout.reference_pulse_shape,
            readout.reference_pulse_sample_width.to_value("ns"),
            (1 / readout.sampling_rate).to_value("ns"),
            self.window_width.tel[telid],
            self.window_shift.tel[telid],
        )

    def __call__(self, waveforms, telid, selected_gain_channel):
        peak_index = waveforms.argmax(axis=-1).astype(np.int64)
        charge, peak_time = extract_around_peak(
            waveforms,
            peak_index,
            self.window_width.tel[telid],
            self.window_shift.tel[telid],
            self.sampling_rate_ghz[telid],
        )
        if self.apply_integration_correction.tel[telid]:
            charge *= self._calculate_correction(telid=telid)[selected_gain_channel]
        return DL1CameraContainer(image=charge, peak_time=peak_time, is_valid=True)


class SlidingWindowMaxSum(ImageExtractor):
    """
    Sliding window extractor that maximizes the signal in window_width consecutive slices.
    """

    window_width = IntTelescopeParameter(
        default_value=7, help="Define the width of the integration window"
    ).tag(config=True)

    apply_integration_correction = BoolTelescopeParameter(
        default_value=True, help="Apply the integration window correction"
    ).tag(config=True)

    @lru_cache(maxsize=128)
    def _calculate_correction(self, telid):
        """
        Calculate the correction for the extracted charge such that the value
        returned would equal 1 for a noise-less unit pulse.

        This method is decorated with @lru_cache to ensure it is only
        calculated once per telescope.

        The same procedure as for the actual SlidingWindowMaxSum extractor is used, but
        on the reference pulse_shape (that is also more finely binned)

        Parameters
        ----------
        telid : int

        Returns
        -------
        correction : ndarray
        The correction to apply to an extracted charge using this ImageExtractor
        Has size n_channels, as a different correction value might be required
        for different gain channels.
        """

        readout = self.subarray.tel[telid].camera.readout

        # compute the number of slices to integrate in the pulse template
        width_shape = int(
            round(
                (
                    self.window_width.tel[telid]
                    / readout.sampling_rate
                    / readout.reference_pulse_sample_width
                )
                .to("")
                .value
            )
        )

        n_channels = len(readout.reference_pulse_shape)
        correction = np.ones(n_channels, dtype=np.float64)
        for ichannel, pulse_shape in enumerate(readout.reference_pulse_shape):

            # apply the same method as sliding window to find the highest sum
            cwf = np.cumsum(pulse_shape)
            # add zero at the begining so it is easier to substract the two arrays later
            cwf = np.concatenate((np.zeros(1), cwf))
            sums = cwf[width_shape:] - cwf[:-width_shape]
            maxsum = np.max(sums)
            correction[ichannel] = np.sum(pulse_shape) / maxsum

        return correction

    def __call__(self, waveforms, telid, selected_gain_channel):
        charge, peak_time = extract_sliding_window(
            waveforms, self.window_width.tel[telid], self.sampling_rate_ghz[telid]
        )
        if self.apply_integration_correction.tel[telid]:
            charge *= self._calculate_correction(telid=telid)[selected_gain_channel]
        return DL1CameraContainer(image=charge, peak_time=peak_time, is_valid=True)


class NeighborPeakWindowSum(ImageExtractor):
    """
    Extractor which sums in a window about the
    peak defined by the wavefroms in neighboring pixels.
    """

    window_width = IntTelescopeParameter(
        default_value=7, help="Define the width of the integration window"
    ).tag(config=True)

    window_shift = IntTelescopeParameter(
        default_value=3,
        help="Define the shift of the integration window "
        "from the peak_index (peak_index - shift)",
    ).tag(config=True)

    lwt = IntTelescopeParameter(
        default_value=0,
        help="Weight of the local pixel (0: peak from neighbors only, "
        "1: local pixel counts as much as any neighbor)",
    ).tag(config=True)

    apply_integration_correction = BoolTelescopeParameter(
        default_value=True, help="Apply the integration window correction"
    ).tag(config=True)

    @lru_cache(maxsize=128)
    def _calculate_correction(self, telid):
        """
        Calculate the correction for the extracted change such that the value
        returned would equal 1 for a noise-less unit pulse.

        This method is decorated with @lru_cache to ensure it is only
        calculated once per telescope.

        Parameters
        ----------
        telid : int

        Returns
        -------
        correction : ndarray
        The correction to apply to an extracted charge using this ImageExtractor
        Has size n_channels, as a different correction value might be required
        for different gain channels.
        """
        readout = self.subarray.tel[telid].camera.readout
        return integration_correction(
            readout.reference_pulse_shape,
            readout.reference_pulse_sample_width.to_value("ns"),
            (1 / readout.sampling_rate).to_value("ns"),
            self.window_width.tel[telid],
            self.window_shift.tel[telid],
        )

    def __call__(self, waveforms, telid, selected_gain_channel):
        neighbors = self.subarray.tel[telid].camera.geometry.neighbor_matrix_sparse
        average_wfs = neighbor_average_waveform(
            waveforms,
            neighbors_indices=neighbors.indices,
            neighbors_indptr=neighbors.indptr,
            lwt=self.lwt.tel[telid],
        )
        peak_index = average_wfs.argmax(axis=-1)
        charge, peak_time = extract_around_peak(
            waveforms,
            peak_index,
            self.window_width.tel[telid],
            self.window_shift.tel[telid],
            self.sampling_rate_ghz[telid],
        )
        if self.apply_integration_correction.tel[telid]:
            charge *= self._calculate_correction(telid=telid)[selected_gain_channel]
        return DL1CameraContainer(image=charge, peak_time=peak_time, is_valid=True)


class BaselineSubtractedNeighborPeakWindowSum(NeighborPeakWindowSum):
    """
    Extractor that first subtracts the baseline before summing in a
    window about the peak defined by the wavefroms in neighboring pixels.
    """

    baseline_start = Int(0, help="Start sample for baseline estimation").tag(
        config=True
    )
    baseline_end = Int(10, help="End sample for baseline estimation").tag(config=True)

    def __call__(self, waveforms, telid, selected_gain_channel):
        baseline_corrected = subtract_baseline(
            waveforms, self.baseline_start, self.baseline_end
        )
        return super().__call__(baseline_corrected, telid, selected_gain_channel)


class TwoPassWindowSum(ImageExtractor):
    """Extractor based on [1]_ which integrates the waveform a second time using
    a time-gradient linear fit. This is in particular the version implemented
    in the CTA-MARS analysis pipeline [2]_.

    Notes
    -----

    #. slide a 3-samples window through the waveform, finding max counts sum;
       the range of the sliding is the one allowing extension from 3 to 5;
       add 1 sample on each side and integrate charge in the 5-sample window;
       time is obtained as a charge-weighted average of the sample numbers;
       No information from neighboouring pixels is used.
    #. Preliminary image cleaning via simple tailcut with minimum number
       of core neighbours set at 1,
    #. Only the brightest cluster of pixels is kept.
    #. Parametrize following Hillas approach only if the resulting image has 3
       or more pixels.
    #. Do a linear fit of pulse time vs. distance along major image axis
       (CTA-MARS uses ROOT "robust" fit option,
       aka Least Trimmed Squares, to get rid of far outliers - this should
       be implemented in 'timing_parameters', e.g scipy.stats.siegelslopes).
    #. For all pixels except the core ones in the main island, integrate
       the waveform once more, in a fixed window of 5 samples set at the time
       "predicted" by the linear time fit.
       If the predicted time for a pixel leads to a window outside the readout
       window, then integrate the last (or first) 5 samples.
    #. The result is an image with main-island core pixels calibrated with a
       1st pass and non-core pixels re-calibrated with a 2nd pass.

    References
    ----------
    .. [1] J. Holder et al., Astroparticle Physics, 25, 6, 391 (2006)
    .. [2] https://forge.in2p3.fr/projects/step-by-step-reference-mars-analysis/wiki

    """

    # Get thresholds for core-pixels depending on telescope type.
    # WARNING: default values are not yet optimized
    core_threshold = FloatTelescopeParameter(
        default_value=[
            ("type", "*", 6.0),
            ("type", "LST*", 6.0),
            ("type", "MST*", 8.0),
            ("type", "SST*", 4.0),
        ],
        help="Picture threshold for internal tail-cuts pass",
    ).tag(config=True)

    disable_second_pass = Bool(
        default_value=False,
        help="only run the first pass of the extractor, for debugging purposes",
    ).tag(config=True)

    apply_integration_correction = BoolTelescopeParameter(
        default_value=True, help="Apply the integration window correction"
    ).tag(config=True)

    @lru_cache(maxsize=4096)
    def _calculate_correction(self, telid, width, shift):
        """Obtain the correction for the integration window specified for each
        pixel.

        The TwoPassWindowSum image extractor applies potentially different
        parameters for the integration window to each pixel, depending on the
        position of the peak. It has been decided to apply gain selection
        directly here. For basic definitions look at the documentation of
        `integration_correction`.

        Parameters
        ----------
        telid : int
            Index of the telescope in use.
        width : int
            Width of the integration window in samples
        shift : int
            Window shift to the left of the pulse peak in samples

        Returns
        -------
        correction : ndarray
            Value of the pixel-wise gain-selected integration correction.

        """
        readout = self.subarray.tel[telid].camera.readout
        # Calculate correction of first pixel for both channels
        return integration_correction(
            readout.reference_pulse_shape,
            readout.reference_pulse_sample_width.to_value("ns"),
            (1 / readout.sampling_rate).to_value("ns"),
            width,
            shift,
        )

    def _apply_first_pass(
        self, waveforms, telid
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute step 1.

        Parameters
        ----------
        waveforms : array of size (N_pixels, N_samples)
            DL0-level waveforms of one event.
        telid : int
            Index of the telescope.

        Returns
        -------
        charge : array_like
            Integrated charge per pixel.
            Shape: (n_pix)
        pulse_time : array_like
            Samples in which the waveform peak has been recognized.
            Shape: (n_pix)
        correction : ndarray
            pixel-wise integration correction
        """
        # STEP 1

        # Starting from DL0, the channel is already selected (if more than one)
        # event.dl0.tel[tel_id].waveform object has shape (N_pixels, N_samples)

        # For each pixel, we slide a 3-samples window through the
        # waveform summing each time the ADC counts contained within it.

        peak_search_window_width = 3
        sums = convolve1d(
            waveforms, np.ones(peak_search_window_width), axis=1, mode="nearest"
        )
        # 'sums' has now still shape of (N_pixels, N_samples)
        # each element is the center-sample of each 3-samples sliding window

        # For each pixel, we check where the peak search window encountered
        # the maximum number of ADC counts.
        # We want to stop before the edge of the readout window in order to
        # later extend the search window to a 1+3+1 integration window.
        # Since in 'sums' the peak index corresponds to the center of the
        # search window, we shift it on the right by 2 samples so to get the
        # correspondent sample index in each waveform.
        peak_index = np.argmax(sums[:, 2:-2], axis=1) + 2
        # Now peak_index has the shape of (N_pixels).

        # The final 5-samples window will be 1+3+1, centered on the 3-samples
        # window in which the highest amount of ADC counts has been found
        window_width = peak_search_window_width + 2
        window_shift = 2

        # this function is applied to all pixels together
        charge_1stpass, pulse_time_1stpass = extract_around_peak(
            waveforms,
            peak_index,
            window_width,
            window_shift,
            self.sampling_rate_ghz[telid],
        )

        # Get integration correction factors
        if self.apply_integration_correction.tel[telid]:
            correction = self._calculate_correction(telid, window_width, window_shift)
        else:
            correction = np.ones(waveforms.shape[0])

        return charge_1stpass, pulse_time_1stpass, correction

    def _apply_second_pass(
        self,
        waveforms,
        telid,
        selected_gain_channel,
        charge_1stpass_uncorrected,
        pulse_time_1stpass,
        correction,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Follow steps from 2 to 7.

        Parameters
        ----------
        waveforms : array of shape (N_pixels, N_samples)
            DL0-level waveforms of one event.
        telid : int
            Index of the telescope.
        selected_gain_channel: array of shape (N_channels, N_pixels)
            Array containing the index of the selected gain channel for each
            pixel (0 for low gain, 1 for high gain).
        charge_1stpass_uncorrected : array of shape N_pixels
            Pixel charges reconstructed with the 1st pass, but not corrected.
        pulse_time_1stpass : array of shape N_pixels
            Pixel-wise pulse times reconstructed with the 1st pass.
        correction: array of shape N_pixels
            Charge correction from 1st pass.

        Returns
        -------
        charge : array_like
            Integrated charge per pixel.
            Note that in the case of a very bright full-camera image this can
            coincide the 1st pass information.
            Also in the case of very dim images the 1st pass will be recycled,
            but in this case the resulting image should be discarded
            from further analysis.
            Shape: (n_pix)
        pulse_time : array_like
            Samples in which the waveform peak has been recognized.
            Same specifications as above.
            Shape: (n_pix)
        is_valid: bool
            True=second-pass succeeded, False=second-pass failed, first pass used
        """
        # STEP 2

        # Apply correction to 1st pass charges
        charge_1stpass = charge_1stpass_uncorrected * correction[selected_gain_channel]

        # Set thresholds for core-pixels depending on telescope
        core_th = self.core_threshold.tel[telid]
        # Boundary thresholds will be half of core thresholds.

        # Preliminary image cleaning with simple two-level tail-cut
        camera_geometry = self.subarray.tel[telid].camera.geometry
        mask_clean = tailcuts_clean(
            camera_geometry,
            charge_1stpass,
            picture_thresh=core_th,
            boundary_thresh=core_th / 2,
            keep_isolated_pixels=False,
            min_number_picture_neighbors=1,
        )

        # STEP 3

        # find all islands using this cleaning
        num_islands, labels = number_of_islands(camera_geometry, mask_clean)

        if num_islands > 0:
            # ...find the brightest one
            mask_brightest_island = brightest_island(
                num_islands, labels, charge_1stpass
            )
        else:
            mask_brightest_island = mask_clean

        # for all pixels except the core ones in the main island of the
        # preliminary image, the waveform will be integrated once more (2nd pass)

        mask_2nd_pass = ~mask_brightest_island | (
            mask_brightest_island & (charge_1stpass < core_th)
        )

        # STEP 4

        # if the resulting image has less then 3 pixels
        if np.count_nonzero(mask_brightest_island) < 3:
            # we return the 1st pass information
            return charge_1stpass, pulse_time_1stpass, False

        # otherwise we proceed by parametrizing the image
        camera_geometry_brightest = camera_geometry[mask_brightest_island]
        charge_brightest = charge_1stpass[mask_brightest_island]
        hillas = hillas_parameters(camera_geometry_brightest, charge_brightest)

        # STEP 5

        # linear fit of pulse time vs. distance along major image axis
        # using only the main island surviving the preliminary
        # image cleaning
        timing = timing_parameters(
            geom=camera_geometry_brightest,
            image=charge_brightest,
            peak_time=pulse_time_1stpass[mask_brightest_island],
            hillas_parameters=hillas,
        )

        # If the fit returns nan
        if np.isnan(timing.slope):
            return charge_1stpass, pulse_time_1stpass, False

        # get projected distances along main image axis
        longitude, _ = camera_to_shower_coordinates(
            camera_geometry.pix_x, camera_geometry.pix_y, hillas.x, hillas.y, hillas.psi
        )

        # get the predicted times as a linear relation
        predicted_pulse_times = (
            timing.slope * longitude[mask_2nd_pass] + timing.intercept
        )

        # Convert time in ns to sample index using the sampling rate from
        # the readout.
        # Approximate the value obtained to nearest integer, then cast to
        # int64 otherwise 'extract_around_peak' complains.

        predicted_peaks = np.rint(
            predicted_pulse_times.value * self.sampling_rate_ghz[telid]
        ).astype(np.int64)

        # Due to the fit these peak indexes can lead to an integration window
        # outside the readout window, so in the next step we check for this.

        # STEP 6

        # select all pixels except the core ones in the main island
        waveforms_to_repass = waveforms[mask_2nd_pass]

        # Build 'width' and 'shift' arrays that adapt on the position of the
        # window along each waveform

        # As before we will integrate the charge in a 5-sample window centered
        # on the peak
        window_width_default = 5
        window_shift_default = 2

        # first we find where the integration window edges WOULD BE
        integration_windows_start = predicted_peaks - window_shift_default
        integration_windows_end = integration_windows_start + window_width_default

        # then we define 2 possible edge cases
        # the predicted integration window falls before the readout window
        integration_before_readout = integration_windows_start < 0
        # or after
        integration_after_readout = integration_windows_end > (
            waveforms_to_repass.shape[1] - 1
        )

        # If the resulting 5-samples window falls before the readout
        # window we take the first 5 samples
        window_width_before = 5
        window_shift_before = 0

        # If the resulting 5-samples window falls after the readout
        # window we take the last 5 samples
        window_width_after = 6
        window_shift_after = 4

        # put all values of widths and shifts for 2nd pass pixels together
        window_widths = np.full(waveforms_to_repass.shape[0], window_width_default)
        window_widths[integration_before_readout] = window_width_before
        window_widths[integration_after_readout] = window_width_after
        window_shifts = np.full(waveforms_to_repass.shape[0], window_shift_default)
        window_shifts[integration_before_readout] = window_shift_before
        window_shifts[integration_after_readout] = window_shift_after

        # Now we have to (re)define the pathological predicted times for which
        # - either the peak itself falls outside of the readout window
        # - or is within the first or last 2 samples (so that at least 1 sample
        # of the integration window is outside of the readout window)
        # We place them at the first or last sample, so the special window
        # widhts and shifts that we defined earlier put the integration window
        # for these 2 cases either in the first 5 samples or the last

        # set sample to 0 (beginning of the waveform)
        # if predicted time falls before
        # but also if it's so near the edge that the integration window falls
        # outside
        predicted_peaks[predicted_peaks < 2] = 0

        # set sample to max-1 (first sample has index 0)
        # if predicted time falls after
        predicted_peaks[predicted_peaks > (waveforms_to_repass.shape[1] - 3)] = (
            waveforms_to_repass.shape[1] - 1
        )

        # re-calibrate 2nd pass pixels using the fixed 5-samples window
        reintegrated_charge, reestimated_pulse_times = extract_around_peak(
            waveforms_to_repass,
            predicted_peaks,
            window_widths,
            window_shifts,
            self.sampling_rate_ghz[telid],
        )

        if self.apply_integration_correction.tel[telid]:
            # Modify integration correction factors only for non-core pixels
            # now we compute 3 corrections for the default, before, and after cases:
            correction = self._calculate_correction(
                telid, window_width_default, window_shift_default
            )[selected_gain_channel][mask_2nd_pass]

            correction_before = self._calculate_correction(
                telid, window_width_before, window_shift_before
            )[selected_gain_channel][mask_2nd_pass]

            correction_after = self._calculate_correction(
                telid, window_width_after, window_shift_after
            )[selected_gain_channel][mask_2nd_pass]

            correction[integration_before_readout] = correction_before[
                integration_before_readout
            ]
            correction[integration_after_readout] = correction_after[
                integration_after_readout
            ]

            reintegrated_charge *= correction

        # STEP 7

        # Combine in the final output with,
        # - core pixels from the main cluster
        # - rest of the pixels which have been passed a second time

        # Start from a copy of the 1st pass charge
        charge_2ndpass = charge_1stpass.copy()
        # Overwrite the charges of pixels marked for second pass
        # leaving untouched the core pixels of the main island
        # from the preliminary (cleaned) image
        charge_2ndpass[mask_2nd_pass] = reintegrated_charge

        # Same approach for the pulse times
        pulse_time_2ndpass = pulse_time_1stpass.copy()
        pulse_time_2ndpass[mask_2nd_pass] = reestimated_pulse_times

        return charge_2ndpass, pulse_time_2ndpass, True

    def __call__(self, waveforms, telid, selected_gain_channel):
        charge1, pulse_time1, correction1 = self._apply_first_pass(waveforms, telid)

        # FIXME: properly make sure that output is 32Bit instead of downcasting here
        if self.disable_second_pass:
            return DL1CameraContainer(
                image=(charge1 * correction1[selected_gain_channel]).astype("float32"),
                peak_time=pulse_time1.astype("float32"),
                is_valid=True,
            )

        charge2, pulse_time2, is_valid = self._apply_second_pass(
            waveforms, telid, selected_gain_channel, charge1, pulse_time1, correction1
        )
        # FIXME: properly make sure that output is 32Bit instead of downcasting here
        return DL1CameraContainer(
            image=charge2.astype("float32"),
            peak_time=pulse_time2.astype("float32"),
            is_valid=is_valid,
        )
