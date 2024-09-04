"""
Charge extraction algorithms to reduce the image to one value per pixel
"""

__all__ = [
    "ImageExtractor",
    "FlashCamExtractor",
    "FullWaveformSum",
    "FixedWindowSum",
    "GlobalPeakWindowSum",
    "LocalPeakWindowSum",
    "SlidingWindowMaxSum",
    "NeighborPeakWindowSum",
    "BaselineSubtractedNeighborPeakWindowSum",
    "TwoPassWindowSum",
    "VarianceExtractor",
    "extract_around_peak",
    "extract_sliding_window",
    "neighbor_average_maximum",
    "subtract_baseline",
    "integration_correction",
]


from abc import abstractmethod
from collections.abc import Callable
from functools import lru_cache

import astropy.units as u
import numpy as np
import numpy.typing as npt
import scipy.stats
from numba import float32, float64, guvectorize, int64, njit, prange
from scipy.ndimage import convolve1d
from traitlets import Bool, Int

from ctapipe.containers import (
    DL1CameraContainer,
    VarianceType,
)
from ctapipe.core import TelescopeComponent
from ctapipe.core.traits import (
    BoolTelescopeParameter,
    ComponentName,
    FloatTelescopeParameter,
    IntTelescopeParameter,
)
from ctapipe.instrument import CameraDescription

from .cleaning import tailcuts_clean
from .hillas import camera_to_shower_coordinates, hillas_parameters
from .invalid_pixels import InvalidPixelHandler
from .morphology import brightest_island, number_of_islands
from .statistics import arg_n_largest
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
        Shape: (n_channels, n_pix, n_samples)
    peak_index : ndarray or int
        Peak index for each pixel.
    width : ndarray or int
        Window size of integration window for each pixel.
    shift : ndarray or int
        Shift of the integration window from the peak_index.
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
        Shape: (n_channels, n_pix)
    peak_time: ndarray
        Extracted peak time.
        Shape: (n_channels, n_pix)

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
    cache=True,
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
        Shape: (n_channels, n_pix, n_samples)
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
        Shape: (n_channels, n_pix)
    peak_time: ndarray
        Extracted peak time.
        Shape: (n_channels, n_pix)

    """

    # first find the cumulative waveform
    cwf = np.cumsum(waveforms)
    # add zero at the beginning so it is easier to subtract the two arrays later
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


@njit(cache=True)
def neighbor_average_maximum(
    waveforms, neighbors_indices, neighbors_indptr, local_weight, broken_pixels
):
    """
    Obtain the average waveform built from the neighbors of each pixel

    Parameters
    ----------
    waveforms : ndarray
        Waveforms stored in a numpy array.
        Shape: (n_channels, n_pix, n_samples)
    neighbors_indices : ndarray
        indices of a scipy csr sparse matrix of neighbors, i.e.
        ``ctapipe.instrument.CameraGeometry.neighbor_matrix_sparse.indices``.
    neighbors_indptr : ndarray
        indptr of a scipy csr sparse matrix of neighbors, i.e.
        ``ctapipe.instrument.CameraGeometry.neighbor_matrix_sparse.indptr``.
    local_weight : int
        Weight of the local pixel (0: peak from neighbors only,
        1: local pixel counts as much as any neighbor)
    broken_pixels : ndarray
        Mask of broken pixels. Broken pixels are ignored in the sum over the
        neighbors.
        Shape: (n_channels, n_pix)

    Returns
    -------
    average_wf : ndarray
        Average of neighbor waveforms for each pixel.
        Shape: (n_channels, n_pix)

    """

    n_channels, n_pixels, _ = waveforms.shape
    indptr = neighbors_indptr
    indices = neighbors_indices

    # initialize to waveforms weighted with local_weight
    # so the value of the pixel itself is already taken into account
    peak_pos = np.empty((n_channels, n_pixels), dtype=np.int64)

    for ichannel in prange(n_channels):
        for pixel in prange(n_pixels):
            average = waveforms[ichannel, pixel] * local_weight
            neighbors = indices[indptr[pixel] : indptr[pixel + 1]]

            for neighbor in neighbors:
                if broken_pixels[ichannel, neighbor]:
                    continue
                average += waveforms[ichannel][neighbor]

            peak_pos[ichannel, pixel] = np.argmax(average)

    return peak_pos


def subtract_baseline(waveforms, baseline_start, baseline_end):
    """
    Subtracts the waveform baseline, estimated as the mean waveform value
    in the interval [baseline_start:baseline_end]

    Parameters
    ----------
    waveforms : ndarray
        Waveforms stored in a numpy array.
        Shape: (n_channels, n_pix, n_samples)
    baseline_start : int
        Sample where the baseline window starts
    baseline_end : int
        Sample where the baseline window ends

    Returns
    -------
    baseline_corrected : ndarray
        Waveform with the baseline subtracted
        Shape: (n_channels, n_pix, n_samples)
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
            tel_id: telescope.camera.readout.sampling_rate.to_value("GHz")
            for tel_id, telescope in subarray.tel.items()
        }

    def _calculate_correction(self, tel_id):
        """
        Calculate the correction for the extracted charge such that the value
        returned would equal 1 for a noise-less unit pulse. `ImageExtractor` types
        calculating corrections need to overwrite this method.

        This method should be decorated with @lru_cache to ensure it is only
        calculated once per telescope.

        Parameters
        ----------
        tel_id : int

        Returns
        -------
        correction : ndarray
        The correction to apply to an extracted charge using this ImageExtractor
        Has size n_channels, as a different correction value might be required
        for different gain channels.
        """
        pass

    @staticmethod
    def _apply_correction(charge, correction, selected_gain_channel):
        """
        Helper function for applying the integration correction for certain `ImageExtractor`s.
        """
        if selected_gain_channel is None:
            return (charge * correction[:, np.newaxis]).astype(charge.dtype)
        return (charge * correction[selected_gain_channel]).astype(charge.dtype)

    @abstractmethod
    def __call__(
        self, waveforms, tel_id, selected_gain_channel, broken_pixels
    ) -> DL1CameraContainer:
        """
        Call the relevant functions to fully extract the charge and time
        for the particular extractor.

        Parameters
        ----------
        waveforms : ndarray
            Waveforms stored in a numpy array of shape
            (n_channels, n_pix, n_samples).
        tel_id : int
            The telescope id. Used to obtain to correct traitlet configuration
            and instrument properties
        selected_gain_channel : ndarray
            The channel selected in the gain selection, per pixel. Required in
            some cases to calculate the correct correction for the charge
            extraction.
        broken_pixels : ndarray
            Mask of broken pixels used for certain `ImageExtractor` types.
            Shape: (n_channels, n_pix)

        Returns
        -------
        DL1CameraContainer:
            extracted images and validity flags
        """


class FullWaveformSum(ImageExtractor):
    """
    Extractor that sums the entire waveform.
    """

    def __call__(
        self, waveforms, tel_id, selected_gain_channel, broken_pixels
    ) -> DL1CameraContainer:
        charge, peak_time = extract_around_peak(
            waveforms, 0, waveforms.shape[-1], 0, self.sampling_rate_ghz[tel_id]
        )

        # reduce dimensions for gain selected data to (n_pixels, )
        if selected_gain_channel is not None:
            charge = charge[0]
            peak_time = peak_time[0]

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
    def _calculate_correction(self, tel_id):
        readout = self.subarray.tel[tel_id].camera.readout
        return integration_correction(
            readout.reference_pulse_shape,
            readout.reference_pulse_sample_width.to_value("ns"),
            (1 / readout.sampling_rate).to_value("ns"),
            self.window_width.tel[tel_id],
            self.window_shift.tel[tel_id],
        )

    def __call__(
        self, waveforms, tel_id, selected_gain_channel, broken_pixels
    ) -> DL1CameraContainer:
        charge, peak_time = extract_around_peak(
            waveforms,
            self.peak_index.tel[tel_id],
            self.window_width.tel[tel_id],
            self.window_shift.tel[tel_id],
            self.sampling_rate_ghz[tel_id],
        )
        if self.apply_integration_correction.tel[tel_id]:
            correction = self._calculate_correction(tel_id=tel_id)
            charge = self._apply_correction(charge, correction, selected_gain_channel)

        # reduce dimensions for gain selected data to (n_pixels, )
        if selected_gain_channel is not None:
            charge = charge[0]
            peak_time = peak_time[0]

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
    def _calculate_correction(self, tel_id):
        readout = self.subarray.tel[tel_id].camera.readout
        return integration_correction(
            readout.reference_pulse_shape,
            readout.reference_pulse_sample_width.to_value("ns"),
            (1 / readout.sampling_rate).to_value("ns"),
            self.window_width.tel[tel_id],
            self.window_shift.tel[tel_id],
        )

    def __call__(
        self, waveforms, tel_id, selected_gain_channel, broken_pixels
    ) -> DL1CameraContainer:
        if self.pixel_fraction.tel[tel_id] == 1.0:
            # average over pixels then argmax over samples
            peak_index = waveforms.mean(
                axis=-2, where=~broken_pixels[..., np.newaxis]
            ).argmax(axis=-1)
        else:
            n_pixels = int(self.pixel_fraction.tel[tel_id] * waveforms.shape[-2])
            brightest = arg_n_largest(
                n_pixels,
                waveforms.max(
                    axis=-1, where=~broken_pixels[..., np.newaxis], initial=-np.inf
                ),
            )

            # average over brightest pixels then argmax over samples
            peak_index = (
                waveforms[:, brightest][:, 0, ...].mean(axis=-2).argmax(axis=-1)
            )

        charge, peak_time = extract_around_peak(
            waveforms,
            peak_index[:, np.newaxis],
            self.window_width.tel[tel_id],
            self.window_shift.tel[tel_id],
            self.sampling_rate_ghz[tel_id],
        )
        if self.apply_integration_correction.tel[tel_id]:
            correction = self._calculate_correction(tel_id=tel_id)
            charge = self._apply_correction(charge, correction, selected_gain_channel)

        # reduce dimensions for gain selected data to (n_pixels, )
        if selected_gain_channel is not None:
            charge = charge[0]
            peak_time = peak_time[0]

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
    def _calculate_correction(self, tel_id):
        readout = self.subarray.tel[tel_id].camera.readout
        return integration_correction(
            readout.reference_pulse_shape,
            readout.reference_pulse_sample_width.to_value("ns"),
            (1 / readout.sampling_rate).to_value("ns"),
            self.window_width.tel[tel_id],
            self.window_shift.tel[tel_id],
        )

    def __call__(
        self, waveforms, tel_id, selected_gain_channel, broken_pixels
    ) -> DL1CameraContainer:
        peak_index = waveforms.argmax(axis=-1).astype(np.int64)
        charge, peak_time = extract_around_peak(
            waveforms,
            peak_index,
            self.window_width.tel[tel_id],
            self.window_shift.tel[tel_id],
            self.sampling_rate_ghz[tel_id],
        )
        if self.apply_integration_correction.tel[tel_id]:
            correction = self._calculate_correction(tel_id=tel_id)
            charge = self._apply_correction(charge, correction, selected_gain_channel)

        # reduce dimensions for gain selected data to (n_pixels, )
        if selected_gain_channel is not None:
            charge = charge[0]
            peak_time = peak_time[0]

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
    def _calculate_correction(self, tel_id):
        readout = self.subarray.tel[tel_id].camera.readout

        # compute the number of slices to integrate in the pulse template
        width_shape = int(
            round(
                (
                    self.window_width.tel[tel_id]
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
            # add zero at the beginning so it is easier to subtract the two arrays later
            cwf = np.concatenate((np.zeros(1), cwf))
            sums = cwf[width_shape:] - cwf[:-width_shape]
            maxsum = np.max(sums)
            correction[ichannel] = np.sum(pulse_shape) / maxsum

        return correction

    def __call__(
        self, waveforms, tel_id, selected_gain_channel, broken_pixels
    ) -> DL1CameraContainer:
        charge, peak_time = extract_sliding_window(
            waveforms, self.window_width.tel[tel_id], self.sampling_rate_ghz[tel_id]
        )

        if self.apply_integration_correction.tel[tel_id]:
            correction = self._calculate_correction(tel_id=tel_id)
            charge = self._apply_correction(charge, correction, selected_gain_channel)

        # reduce dimensions for gain selected data to (n_pixels, )
        if selected_gain_channel is not None:
            charge = charge[0]
            peak_time = peak_time[0]

        return DL1CameraContainer(image=charge, peak_time=peak_time, is_valid=True)


class NeighborPeakWindowSum(ImageExtractor):
    """
    Extractor which sums in a window about the
    peak defined by the waveforms in neighboring pixels.
    """

    window_width = IntTelescopeParameter(
        default_value=7, help="Define the width of the integration window"
    ).tag(config=True)

    window_shift = IntTelescopeParameter(
        default_value=3,
        help="Define the shift of the integration window "
        "from the peak_index (peak_index - shift)",
    ).tag(config=True)

    local_weight = IntTelescopeParameter(
        default_value=0,
        help="Weight of the local pixel (0: peak from neighbors only, "
        "1: local pixel counts as much as any neighbor)",
    ).tag(config=True)

    apply_integration_correction = BoolTelescopeParameter(
        default_value=True, help="Apply the integration window correction"
    ).tag(config=True)

    @lru_cache(maxsize=128)
    def _calculate_correction(self, tel_id):
        readout = self.subarray.tel[tel_id].camera.readout
        return integration_correction(
            readout.reference_pulse_shape,
            readout.reference_pulse_sample_width.to_value("ns"),
            (1 / readout.sampling_rate).to_value("ns"),
            self.window_width.tel[tel_id],
            self.window_shift.tel[tel_id],
        )

    def __call__(
        self, waveforms, tel_id, selected_gain_channel, broken_pixels
    ) -> DL1CameraContainer:
        neighbors = self.subarray.tel[tel_id].camera.geometry.neighbor_matrix_sparse
        peak_index = neighbor_average_maximum(
            waveforms,
            neighbors_indices=neighbors.indices,
            neighbors_indptr=neighbors.indptr,
            local_weight=self.local_weight.tel[tel_id],
            broken_pixels=broken_pixels,
        )
        charge, peak_time = extract_around_peak(
            waveforms,
            peak_index,
            self.window_width.tel[tel_id],
            self.window_shift.tel[tel_id],
            self.sampling_rate_ghz[tel_id],
        )

        if self.apply_integration_correction.tel[tel_id]:
            correction = self._calculate_correction(tel_id=tel_id)
            charge = self._apply_correction(charge, correction, selected_gain_channel)

        # reduce dimensions for gain selected data to (n_pixels, )
        if selected_gain_channel is not None:
            charge = charge[0]
            peak_time = peak_time[0]

        return DL1CameraContainer(image=charge, peak_time=peak_time, is_valid=True)


class BaselineSubtractedNeighborPeakWindowSum(NeighborPeakWindowSum):
    """
    Extractor that first subtracts the baseline before summing in a
    window about the peak defined by the waveforms in neighboring pixels.
    """

    baseline_start = Int(0, help="Start sample for baseline estimation").tag(
        config=True
    )
    baseline_end = Int(10, help="End sample for baseline estimation").tag(config=True)

    def __call__(
        self, waveforms, tel_id, selected_gain_channel, broken_pixels
    ) -> DL1CameraContainer:
        baseline_corrected = subtract_baseline(
            waveforms, self.baseline_start, self.baseline_end
        )
        return super().__call__(
            baseline_corrected, tel_id, selected_gain_channel, broken_pixels
        )


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
       No information from neighbouring pixels is used.
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

    invalid_pixel_handler_type = ComponentName(
        InvalidPixelHandler,
        default_value="NeighborAverage",
        help="Name of the InvalidPixelHandler to apply in the first pass.",
        allow_none=True,
    ).tag(config=True)

    def __init__(self, subarray, **kwargs):
        super().__init__(subarray=subarray, **kwargs)
        self.invalid_pixel_handler = None
        if self.invalid_pixel_handler_type is not None:
            self.invalid_pixel_handler = InvalidPixelHandler.from_name(
                self.invalid_pixel_handler_type,
                self.subarray,
                parent=self,
            )

    @lru_cache(maxsize=4096)
    def _calculate_correction(self, tel_id, width, shift):
        """Obtain the correction for the integration window specified for each
        pixel.

        The TwoPassWindowSum image extractor applies potentially different
        parameters for the integration window to each pixel, depending on the
        position of the peak. It has been decided to apply gain selection
        directly here. For basic definitions look at the documentation of
        `integration_correction`.

        Parameters
        ----------
        tel_id : int
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
        readout = self.subarray.tel[tel_id].camera.readout
        # Calculate correction of first pixel for both channels
        return integration_correction(
            readout.reference_pulse_shape,
            readout.reference_pulse_sample_width.to_value("ns"),
            (1 / readout.sampling_rate).to_value("ns"),
            width,
            shift,
        )

    def _apply_first_pass(
        self, waveforms, tel_id
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute step 1.

        Parameters
        ----------
        waveforms : array of size (N_pixels, N_samples)
            DL0-level waveforms of one event.
        tel_id : int
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
            self.sampling_rate_ghz[tel_id],
        )

        # Get integration correction factors
        if self.apply_integration_correction.tel[tel_id]:
            correction = self._calculate_correction(tel_id, window_width, window_shift)
        else:
            correction = np.ones(waveforms.shape[0])

        return charge_1stpass, pulse_time_1stpass, correction

    def _apply_second_pass(
        self,
        waveforms,
        tel_id,
        selected_gain_channel,
        charge_1stpass_uncorrected,
        pulse_time_1stpass,
        correction,
        broken_pixels,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Follow steps from 2 to 7.

        Parameters
        ----------
        waveforms : array of shape (N_pixels, N_samples)
            DL0-level waveforms of one event.
        tel_id : int
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

        camera_geometry = self.subarray.tel[tel_id].camera.geometry
        if self.invalid_pixel_handler is not None:
            charge_1stpass, pulse_time_1stpass = self.invalid_pixel_handler(
                tel_id,
                charge_1stpass,
                pulse_time_1stpass,
                broken_pixels,
            )

        # Set thresholds for core-pixels depending on telescope
        core_th = self.core_threshold.tel[tel_id]
        # Boundary thresholds will be half of core thresholds.

        # Preliminary image cleaning with simple two-level tail-cut
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
        n_islands, labels = number_of_islands(camera_geometry, mask_clean)

        if n_islands > 0:
            # ...find the brightest one
            mask_brightest_island = brightest_island(n_islands, labels, charge_1stpass)
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
            predicted_pulse_times.value * self.sampling_rate_ghz[tel_id]
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
            self.sampling_rate_ghz[tel_id],
        )

        if self.apply_integration_correction.tel[tel_id]:
            # Modify integration correction factors only for non-core pixels
            # now we compute 3 corrections for the default, before, and after cases:
            correction = self._calculate_correction(
                tel_id, window_width_default, window_shift_default
            )[selected_gain_channel][mask_2nd_pass]

            correction_before = self._calculate_correction(
                tel_id, window_width_before, window_shift_before
            )[selected_gain_channel][mask_2nd_pass]

            correction_after = self._calculate_correction(
                tel_id, window_width_after, window_shift_after
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

    def __call__(
        self, waveforms, tel_id, selected_gain_channel, broken_pixels
    ) -> DL1CameraContainer:
        if waveforms.shape[-3] != 1:
            raise AttributeError(
                "The data needs to be gain selected to use the TwoPassWindowSum."
            )
        waveforms = waveforms[0, :, :]

        charge1, pulse_time1, correction1 = self._apply_first_pass(waveforms, tel_id)

        # FIXME: properly make sure that output is 32Bit instead of downcasting here
        if self.disable_second_pass:
            return DL1CameraContainer(
                image=(charge1 * correction1[selected_gain_channel]).astype("float32"),
                peak_time=pulse_time1.astype("float32"),
                is_valid=True,
            )

        charge2, pulse_time2, is_valid = self._apply_second_pass(
            waveforms,
            tel_id,
            selected_gain_channel,
            charge1,
            pulse_time1,
            correction1,
            broken_pixels,
        )
        # FIXME: properly make sure that output is 32Bit instead of downcasting here
        return DL1CameraContainer(
            image=charge2.astype("float32"),
            peak_time=pulse_time2.astype("float32"),
            is_valid=is_valid,
        )


class VarianceExtractor(ImageExtractor):
    """Calculate the variance over samples in each waveform."""

    def __call__(
        self, waveforms, tel_id, selected_gain_channel, broken_pixels
    ) -> DL1CameraContainer:
        container = DL1CameraContainer(
            image=np.nanvar(waveforms, dtype="float32", axis=2),
        )
        container.meta["ExtractionMethod"] = str(VarianceType.WAVEFORM)
        return container


def deconvolution_parameters(
    camera: CameraDescription,
    upsampling: int,
    window_width: int,
    window_shift: int,
    leading_edge_timing: bool,
    leading_edge_rel_descend_limit: float,
    time_profile_pdf: None | Callable[[npt.ArrayLike], npt.ArrayLike] = None,
) -> tuple[list[float], list[float], list[float]]:
    """
    Estimates deconvolution and recalibration parameters from the camera's reference
    single-p.e. pulse shape for the given configuration of FlashCamExtractor.

    Parameters
    ----------
    camera : CameraDescription
        Description of the target camera.
    upsampling : int
        Upsampling factor (>= 1); see also `deconvolve(...)`.
    window_width : int
        Integration window width (>= 1); see also `extract_around_peak(...)`.
    window_shift : int
        Shift of the integration window relative to the peak; see also
        `extract_around_peak(...)`.
    leading_edge_timing : bool
        Whether time calculation will be done on the leading edge.
    leading_edge_rel_descend_limit : float
        If leading edge timing is used, the fraction of the peak value down to which samples are accumulated in the
        centroid calculation.
    time_profile_pdf : callable or None
        PDF of the assumed effective Cherenkov time profile to assume when
        calculating the gain loss; takes nanoseconds as arguments and returns
        probability density (with mode at ~0 ns); default: None (assume
        instantaneous pulse).

    Returns
    -------
    pole_zeros : list of floats
        Pole-zero parameter for each channel to be passed to `deconvolve(...)`.
    gain_losses : list of floats
        Gain loss of each channel that needs to be corrected after deconvolution.
    time_shifts_nsec : list of floats
        Timing shift of each channel that needs to be corrected after deconvolution.
    leading_edge_shifts : list of floats
        Offset of the leading edge peak w.r.t. the deconvolved peak.
    """
    if upsampling < 1:
        raise ValueError(f"upsampling must be > 0, got {upsampling}")
    if window_width < 1:
        raise ValueError(f"window_width must be > 0, got {window_width}")

    ref_pulse_shapes = camera.readout.reference_pulse_shape
    ref_sample_width_nsec = camera.readout.reference_pulse_sample_width.to_value(u.ns)
    camera_sample_width_nsec = 1.0 / camera.readout.sampling_rate.to_value(u.GHz)

    if camera_sample_width_nsec < ref_sample_width_nsec:
        raise ValueError(
            f"Reference pulse sampling time (got {ref_sample_width_nsec} ns) must be equal to or shorter than the "
            f"camera sampling time (got {camera_sample_width_nsec} ns); need a reference single p.e. pulse shape with "
            "finer sampling!"
        )
    avg_step = int(round(camera_sample_width_nsec / ref_sample_width_nsec))

    pole_zeros = []  # avg. pole-zero deconvolution parameters
    for ref_pulse_shape in ref_pulse_shapes:
        phase_pzs = []
        for phase in range(avg_step):
            x = ref_pulse_shape[phase::avg_step]
            i_min = np.argmin(np.diff(x)) + 1
            phase_pzs.append(x[i_min + 1] / x[i_min])

        if len(phase_pzs) == 0:
            raise ValueError(
                "ref_pulse_shape is malformed - cannot find deconvolution parameter"
            )
        pole_zeros.append(np.mean(phase_pzs))

    gains, shifts, pz2d_shifts = [], [], []  # avg. gains and timing shifts
    for pz, ref_pulse_shape in zip(pole_zeros, ref_pulse_shapes):
        if time_profile_pdf:  # convolve ref_pulse_shape with time profile PDF
            t = (
                np.arange(ref_pulse_shape.size) - ref_pulse_shape.size / 2
            ) * ref_sample_width_nsec
            time_profile = time_profile_pdf(t)
            ref_pulse_shape = np.convolve(ref_pulse_shape, time_profile, "same")

        integral = (
            ref_pulse_shape.sum() * ref_sample_width_nsec / camera_sample_width_nsec
        )
        phase_gains, phase_shifts, phase_pz2d_shifts = [], [], []
        for phase in range(avg_step):
            x = np.atleast_2d(ref_pulse_shape[phase::avg_step])
            y = deconvolve(x, 0.0, upsampling, pz)[0]
            i_max = y.argmax()
            start = i_max - window_shift
            stop = start + window_width
            if start >= 0 and stop <= y.size:
                if leading_edge_timing:
                    d = deconvolve(x, 0.0, upsampling, 1)[0]
                    d_pk_idx = d.argmax()
                    phase_pz2d_shifts.append(i_max - d_pk_idx)
                    time = adaptive_centroid(
                        d, d_pk_idx, leading_edge_rel_descend_limit
                    )
                else:
                    time = i_max

                phase_shifts.append(
                    (time / upsampling * avg_step - ref_pulse_shape.argmax())
                    * ref_sample_width_nsec
                )
                phase_gains.append(y[start:stop].sum() / integral)

        if len(phase_gains) == 0:
            raise ValueError(
                "ref_pulse_shape is malformed - peak is not well contained"
            )
        gains.append(np.mean(phase_gains))
        shifts.append(np.mean(phase_shifts))
        pz2d_shifts.append(np.mean(phase_pz2d_shifts))

    return pole_zeros, gains, shifts, pz2d_shifts


def __filtfilt_fast(signal, filt):
    """
    Apply a linear filter forward and backward to a signal, based on scipy.signal.filtfilt.
    filtfilt has some speed issues (https://github.com/scipy/scipy/issues/17080)
    """
    forward = convolve1d(signal, filt, axis=-1, mode="nearest")
    backward = convolve1d(forward[..., ::-1], filt, axis=-1, mode="nearest")
    return backward[..., ::-1]


def deconvolve(
    waveforms: npt.ArrayLike,
    baselines: npt.ArrayLike,
    upsampling: int,
    pole_zero: float,
) -> np.ndarray:
    """
    Applies pole-zero deconvolution and upsampling to pixel waveforms. Use
    `deconvolution_parameters(...)` to estimate the required `pole_zero` parameter
    for the specific camera model.

    Parameters
    ----------
    waveforms : ndarray
        Waveforms stored in a numpy array.
        Shape: (n_channels, n_pix, n_samples)
    baselines : ndarray or float
        Baseline estimates for each pixel that are subtracted from the waveforms
        before deconvolution.
        Shape: (n_pix, ) or scalar
    upsampling : int
        Upsampling factor to use (>= 1); if > 1, the input waveforms are resampled
        at upsampling times their original sampling rate.
    pole_zero : float
        Deconvolution parameter obtained from `deconvolution_parameters(...)`.

    Returns
    -------
    deconvolved_waveforms : ndarray
        Deconvolved and upsampled waveforms stored in a numpy array.
        Shape: (n_channels, n_pix, upsampling * n_samples)
    """
    deconvolved_waveforms = np.atleast_2d(waveforms) - np.atleast_2d(baselines).T
    deconvolved_waveforms[..., 1:] -= pole_zero * deconvolved_waveforms[..., :-1]
    deconvolved_waveforms[..., 0] = 0

    if upsampling > 1:
        filt = np.ones(upsampling)
        filt_weighted = filt / upsampling
        signal = np.repeat(deconvolved_waveforms, upsampling, axis=-1)
        return __filtfilt_fast(signal, filt_weighted)

    return deconvolved_waveforms


@guvectorize(
    [
        (float32[:], int64, float64, float32[:]),
        (float64[:], int64, float64, float32[:]),
    ],
    "(s),(),()->()",
    nopython=True,
    cache=True,
)
def adaptive_centroid(waveforms, peak_index, rel_descend_limit, centroids):
    """
    Calculates the pulse centroid for all samples down to rel_descend_limit * peak_amplitude.

    The ret argument is required by numpy to create the numpy array which is
    returned. It can be ignored when calling this function.

    Parameters
    ----------
    waveforms : ndarray
        Waveforms stored in a numpy array.
        Shape: (n_channels, n_pix, n_samples)
    peak_index : ndarray or int
        Peak index for each pixel.
    rel_descend_limit : ndarray or float
        Fraction of the peak value down to which samples are accumulated in the centroid calculation.
    centroids : ndarray
        Return argument for ufunc (ignore)
        Returns the peak centroid in units "samples"

    Returns
    -------
    centroids : ndarray
        Peak centroid in units "samples"
    """
    centroids[0] = peak_index  # preload in case of errors

    n_samples = waveforms.size
    if n_samples == 0:
        return

    if (peak_index > (n_samples - 1)) or (peak_index < 0):
        raise ValueError("peak_index must be within the waveform limits")

    peak_amplitude = waveforms[peak_index]
    if peak_amplitude < 0.0:
        return

    descend_limit = rel_descend_limit * peak_amplitude

    sum_ = float64(0.0)
    jsum = float64(0.0)

    j = peak_index
    while j >= 0 and waveforms[j] > descend_limit:
        sum_ += waveforms[j]
        jsum += j * waveforms[j]
        j -= 1
        if j >= 0 and waveforms[j] > peak_amplitude:
            descend_limit = rel_descend_limit * peak_amplitude

    j = peak_index + 1
    while j < n_samples and waveforms[j] > descend_limit:
        sum_ += waveforms[j]
        jsum += j * waveforms[j]
        j += 1
        if j < n_samples and waveforms[j] > peak_amplitude:
            descend_limit = rel_descend_limit * peak_amplitude

    if sum_ != 0.0:
        centroids[0] = jsum / sum_


class FlashCamExtractor(ImageExtractor):
    """
    Image extractor applying signal preprocessing for FlashCam

    The waveforms are first upsampled to achieve one nanosecond sampling (as a default, for the FlashCam).
    A pole-zero deconvolution [1] is then performed to the waveforms to recover the original impulse or narrow
    the resulting pulse due to convolution with detector response. The modified waveform is integrated in a
    window around a peak, which is defined by the neighbors of the pixel. The waveforms are clipped in
    order to reduce the contribution from the afterpulses in the neighbor sum. If leading_edge_timing is
    set to True, the so-called leading edge time is found (with the adaptive_centroid function) instead of the peak
    time.

    This extractor has been optimized for the FlashCam [2].

    [1] Smith, S. W. 1997, The Scientist and Engineer’s Guide to Digital Signal Processing (California Technical
    Publishing)
    [2] FlashCam: a novel Cherenkov telescope camera with continuous signal digitization. CTA Consortium.
    A. Gadola (Zurich U.) et al. DOI: 10.1088/1748-0221/10/01/C01014. Published in: JINST 10 (2015) 01, C01014

    """

    upsampling = IntTelescopeParameter(
        default_value=4, help="Define the upsampling factor for waveforms"
    ).tag(config=True, min=1)

    window_width = IntTelescopeParameter(
        default_value=7, help="Define the width of the integration window"
    ).tag(config=True, min=1)

    window_shift = IntTelescopeParameter(
        default_value=3,
        help="Define the shift of the integration window from the peak_index "
        "(peak_index - shift)",
    ).tag(config=True)

    local_weight = IntTelescopeParameter(
        default_value=0,
        help="Weight of the local pixel (0: peak from neighbors only, "
        "1: local pixel counts as much as any neighbor)",
    ).tag(config=True)

    effective_time_profile_std = FloatTelescopeParameter(
        default_value=2.0,
        help="Effective Cherenkov time profile std. dev. (in nanoseconds) to "
        "assume for calculating the gain correction",
    ).tag(config=True, min=0)

    neighbour_sum_clipping = FloatTelescopeParameter(
        default_value=5.0,
        help="(Soft) clipping level of a pixel's contribution to a neighbour sum "
        "(set to 0 or inf to disable clipping)",
    ).tag(config=True)

    leading_edge_timing = BoolTelescopeParameter(
        default_value=True, help="Calculate leading edge time instead of peak time"
    ).tag(config=True)

    leading_edge_rel_descend_limit = FloatTelescopeParameter(
        default_value=0.05,
        help="Fraction of the peak value down to which samples are accumulated "
        "in the leading edge centroid calculation",
    ).tag(config=True, min=0.0, max=1.0)

    def __init__(self, subarray, **kwargs):
        super().__init__(subarray=subarray, **kwargs)

        self.sampling_rate_ghz = {
            tel_id: telescope.camera.readout.sampling_rate.to_value("GHz")
            for tel_id, telescope in subarray.tel.items()
        }
        self._deconvolution_parameters = {}

    def _get_deconvolution_parameters(self, tel_id):
        if tel_id not in self._deconvolution_parameters:
            tel = self.subarray.tel[tel_id]

            def time_profile_pdf_gen(std_dev: float):
                if std_dev == 0:
                    return None
                return scipy.stats.norm(0.0, std_dev).pdf

            self._deconvolution_parameters[tel_id] = deconvolution_parameters(
                tel.camera,
                self.upsampling.tel[tel_id],
                self.window_width.tel[tel_id],
                self.window_shift.tel[tel_id],
                self.leading_edge_timing.tel[tel_id],
                self.leading_edge_rel_descend_limit.tel[tel_id],
                time_profile_pdf_gen(self.effective_time_profile_std.tel[tel_id]),
            )
        return self._deconvolution_parameters[tel_id]

    @staticmethod
    def clip(x, lo=0.0, hi=np.inf):
        """Applies soft clipping to ±1 and then hard clipping to (lo, hi)."""
        return np.clip(x / (1.0 + np.abs(x)), lo, hi)

    def __call__(
        self, waveforms, tel_id, selected_gain_channel, broken_pixels
    ) -> DL1CameraContainer:
        upsampling = self.upsampling.tel[tel_id]
        integration_window_width = self.window_width.tel[tel_id]
        integration_window_shift = self.window_shift.tel[tel_id]
        neighbour_sum_clipping = self.neighbour_sum_clipping.tel[tel_id]
        leading_edge_timing = self.leading_edge_timing.tel[tel_id]
        leading_edge_rel_descend_limit = self.leading_edge_rel_descend_limit.tel[tel_id]

        pole_zeros, gains, shifts, pz2ds = self._get_deconvolution_parameters(tel_id)
        pz, gain, shift, pz2d = pole_zeros[0], gains[0], shifts[0], pz2ds[0]

        t_waveforms = deconvolve(waveforms, 0.0, upsampling, pz)

        if neighbour_sum_clipping == 0.0 or np.isinf(neighbour_sum_clipping):
            nn_waveforms = t_waveforms
        else:
            nn_waveforms = self.clip(t_waveforms / neighbour_sum_clipping)

        neighbors = self.subarray.tel[tel_id].camera.geometry.neighbor_matrix_sparse
        peak_index = neighbor_average_maximum(
            nn_waveforms,
            neighbors_indices=neighbors.indices,
            neighbors_indptr=neighbors.indptr,
            local_weight=self.local_weight.tel[tel_id],
            broken_pixels=broken_pixels,
        )

        charge, peak_time = extract_around_peak(
            t_waveforms,
            peak_index,
            integration_window_width,
            integration_window_shift,
            self.sampling_rate_ghz[tel_id] * upsampling,
        )

        if leading_edge_timing:
            d_waveforms = deconvolve(waveforms, 0.0, upsampling, 1)

            # correct the offset between leading edge peak and deconvolved peak
            peak_index = np.round(peak_index - pz2d).astype(int)
            n_samples = d_waveforms.shape[-1]
            np.clip(peak_index, 0, n_samples - 1, out=peak_index)
            peak_time = adaptive_centroid(
                d_waveforms, peak_index, leading_edge_rel_descend_limit
            )
            peak_time /= self.sampling_rate_ghz[tel_id] * upsampling

        if gain != 0:
            charge /= gain

        if shift != 0:
            peak_time -= shift

        # reduce dimensions for gain selected data to (n_pixels, )
        if selected_gain_channel is not None:
            charge = charge[0]
            peak_time = peak_time[0]

        return DL1CameraContainer(image=charge, peak_time=peak_time, is_valid=True)
