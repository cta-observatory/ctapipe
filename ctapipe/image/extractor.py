"""
Charge extraction algorithms to reduce the image to one value per pixel
"""

__all__ = [
    "ImageExtractor",
    "FullWaveformSum",
    "FixedWindowSum",
    "GlobalPeakWindowSum",
    "LocalPeakWindowSum",
    "NeighborPeakWindowSum",
    "BaselineSubtractedNeighborPeakWindowSum",
    "TwoPassWindowSum",
    "extract_around_peak",
    "neighbor_average_waveform",
    "subtract_baseline",
    "integration_correction",
]


from abc import abstractmethod
from functools import lru_cache
import numpy as np
from traitlets import Int, Bool
from ctapipe.core.traits import (
    IntTelescopeParameter,
    FloatTelescopeParameter,
    BoolTelescopeParameter,
)
from ctapipe.core import TelescopeComponent
from numba import njit, prange, guvectorize, float64, float32, int64
from scipy.ndimage.filters import convolve1d
from typing import Tuple

from . import number_of_islands, largest_island, tailcuts_clean
from .timing import timing_parameters
from .hillas import hillas_parameters, camera_to_shower_coordinates


@guvectorize(
    [
        (float64[:], int64, int64, int64, float64, float32[:], float32[:]),
        (float32[:], int64, int64, int64, float64, float32[:], float32[:]),
    ],
    "(s),(),(),(),()->(),()",
    nopython=True,
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


@njit(parallel=True)
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
        `ctapipe.instrument.CameraGeometry.neighbor_matrix_sparse.indices`.
    neighbors_indptr : ndarray
        indptr of a scipy csr sparse matrix of neighbors, i.e.
        `ctapipe.instrument.CameraGeometry.neighbor_matrix_sparse.indptr`.
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
    correction = np.ones(n_channels, dtype=np.float)
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

        self.sampling_rate = {
            telid: telescope.camera.readout.sampling_rate.to_value("GHz")
            for telid, telescope in subarray.tel.items()
        }

    @abstractmethod
    def __call__(self, waveforms, telid, selected_gain_channel):
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
        charge : ndarray
            Charge extracted from the waveform in "waveform_units * ns"
            Shape: (n_pix)
        peak_time : ndarray
            Floating point pulse time in each pixel in units "ns"
            Shape: (n_pix)
        """


class FullWaveformSum(ImageExtractor):
    """
    Extractor that sums the entire waveform.
    """

    def __call__(self, waveforms, telid, selected_gain_channel):
        charge, peak_time = extract_around_peak(
            waveforms, 0, waveforms.shape[-1], 0, self.sampling_rate[telid]
        )
        return charge, peak_time


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
            self.sampling_rate[telid],
        )
        if self.apply_integration_correction.tel[telid]:
            charge *= self._calculate_correction(telid=telid)[selected_gain_channel]
        return charge, peak_time


class GlobalPeakWindowSum(ImageExtractor):
    """
    Extractor which sums in a window about the
    peak from the global average waveform.
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
        peak_index = waveforms.mean(axis=-2).argmax(axis=-1)
        charge, peak_time = extract_around_peak(
            waveforms,
            peak_index,
            self.window_width.tel[telid],
            self.window_shift.tel[telid],
            self.sampling_rate[telid],
        )
        if self.apply_integration_correction.tel[telid]:
            charge *= self._calculate_correction(telid=telid)[selected_gain_channel]
        return charge, peak_time


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
        peak_index = waveforms.argmax(axis=-1).astype(np.int)
        charge, peak_time = extract_around_peak(
            waveforms,
            peak_index,
            self.window_width.tel[telid],
            self.window_shift.tel[telid],
            self.sampling_rate[telid],
        )
        if self.apply_integration_correction.tel[telid]:
            charge *= self._calculate_correction(telid=telid)[selected_gain_channel]
        return charge, peak_time


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
            self.sampling_rate[telid],
        )
        if self.apply_integration_correction.tel[telid]:
            charge *= self._calculate_correction(telid=telid)[selected_gain_channel]
        return charge, peak_time


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
    #. Only the biggest cluster of pixels is kept.
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
            self.sampling_rate[telid],
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
        """
        # STEP 2

        # Apply correction to 1st pass charges
        charge_1stpass = charge_1stpass_uncorrected * correction[selected_gain_channel]

        # Set thresholds for core-pixels depending on telescope
        core_th = self.core_threshold.tel[telid]
        # Boundary thresholds will be half of core thresholds.

        # Preliminary image cleaning with simple two-level tail-cut
        camera_geometry = self.subarray.tel[telid].camera.geometry
        mask_1 = tailcuts_clean(
            camera_geometry,
            charge_1stpass,
            picture_thresh=core_th,
            boundary_thresh=core_th / 2,
            keep_isolated_pixels=False,
            min_number_picture_neighbors=1,
        )
        image_1 = charge_1stpass.copy()
        image_1[~mask_1] = 0

        # STEP 3

        # find all islands using this cleaning
        num_islands, labels = number_of_islands(camera_geometry, mask_1)
        if num_islands == 0:
            image_2 = image_1.copy()  # no islands = image unchanged
        else:
            # ...find the biggest one
            mask_biggest = largest_island(labels)
            image_2 = image_1.copy()
            image_2[~mask_biggest] = 0

        # Indexes of pixels that will need the 2nd pass
        non_core_pixels_ids = np.where(image_2 < core_th)[0]
        non_core_pixels_mask = image_2 < core_th

        # STEP 4

        # if the resulting image has less then 3 pixels
        # or there are more than 3 pixels but all contain a number of
        # photoelectrons above the core threshold
        if np.count_nonzero(image_2) < 3:
            # we return the 1st pass information
            # NOTE: In this case, the image was not bright enough!
            # We should label it as "bad and NOT use it"
            return charge_1stpass, pulse_time_1stpass
        elif len(non_core_pixels_ids) == 0:
            # Since all reconstructed charges are above the core threshold,
            # there is no need to perform the 2nd pass.
            # We return the 1st pass information.
            # NOTE: In this case, even if this is 1st pass information,
            # the image is actually very bright! We should label it as "good"!
            return charge_1stpass, pulse_time_1stpass

        # otherwise we proceed by parametrizing the image
        hillas = hillas_parameters(camera_geometry, image_2)

        # STEP 5

        # linear fit of pulse time vs. distance along major image axis
        # using only the main island surviving the preliminary
        # image cleaning
        timing = timing_parameters(camera_geometry, image_2, pulse_time_1stpass, hillas)

        # If the fit returns nan
        if np.isnan(timing.slope):
            return charge_1stpass, pulse_time_1stpass

        # get projected distances along main image axis
        long, _ = camera_to_shower_coordinates(
            camera_geometry.pix_x, camera_geometry.pix_y, hillas.x, hillas.y, hillas.psi
        )

        # get the predicted times as a linear relation
        predicted_pulse_times = (
            timing.slope * long[non_core_pixels_ids] + timing.intercept
        )

        predicted_peaks = np.zeros(len(predicted_pulse_times))

        # Convert time in ns to sample index using the sampling rate from
        # the readout.
        # Approximate the value obtained to nearest integer, then cast to
        # int64 otherwise 'extract_around_peak' complains.
        sampling_rate = self.sampling_rate[telid]
        np.rint(predicted_pulse_times.value * sampling_rate, predicted_peaks)
        predicted_peaks = predicted_peaks.astype(np.int64)

        # Due to the fit these peak indexes can now be also outside of the
        # readout window, so later we check for this.

        # STEP 6

        # select only the waveforms correspondent to the non-core pixels
        # of the main island survived from the 1st pass image cleaning
        non_core_waveforms = waveforms[non_core_pixels_ids]

        # Build 'width' and 'shift' arrays that adapt on the position of the
        # window along each waveform

        # As before we will integrate the charge in a 5-sample window centered
        # on the peak
        window_width_default = 5
        window_shift_default = 2

        # now let's deal with some edge cases: the predicted peak falls before
        # or after the readout window:
        peak_before_window = predicted_peaks < 0
        peak_after_window = predicted_peaks > (non_core_waveforms.shape[1] - 1)

        # BUT, if the resulting 5-samples window falls outside of the readout
        # window then we take the first (or last) 5 samples
        window_width_before = 5
        window_shift_before = 0

        # in the case where the window is after, shift backward
        window_width_after = 5
        window_shift_after = 5

        # and put them together:
        window_widths = np.full(non_core_waveforms.shape[0], window_width_default)
        window_widths[peak_before_window] = window_width_before
        window_widths[peak_after_window] = window_width_after
        window_shifts = np.full(non_core_waveforms.shape[0], window_shift_default)
        window_shifts[peak_before_window] = window_shift_before
        window_shifts[peak_after_window] = window_shift_after

        # Now we can also (re)define the patological predicted times
        # because (we needed them to define the corrispective widths
        # and shifts)
        # set sample to 0 (beginning of the waveform) if predicted time
        # falls before
        predicted_peaks[predicted_peaks < 0] = 0
        # set sample to max-1 (first sample has index 0)
        # if predicted time falls after
        predicted_peaks[predicted_peaks > (waveforms.shape[1] - 1)] = (
            waveforms.shape[1] - 1
        )

        # re-calibrate non-core pixels using the fixed 5-samples window
        charge_no_core, pulse_times_no_core = extract_around_peak(
            non_core_waveforms,
            predicted_peaks,
            window_widths,
            window_shifts,
            self.sampling_rate[telid],
        )

        if self.apply_integration_correction.tel[telid]:
            # Modify integration correction factors only for non-core pixels
            # now we compute 3 corrections for the default, before, and after cases:
            correction = self._calculate_correction(
                telid, window_width_default, window_shift_default
            )[selected_gain_channel][non_core_pixels_mask]

            correction_before = self._calculate_correction(
                telid, window_width_before, window_shift_before
            )[selected_gain_channel][non_core_pixels_mask]

            correction_after = self._calculate_correction(
                telid, window_width_after, window_shift_after
            )[selected_gain_channel][non_core_pixels_mask]

            correction[peak_before_window] = correction_before[peak_before_window]
            correction[peak_after_window] = correction_after[peak_after_window]

            charge_no_core *= correction

        # STEP 7

        # Combine core and non-core pixels in the final output

        # this is the biggest cluster from the cleaned image
        # it contains the core pixels (which we leave untouched)
        # plus possibly some non-core pixels
        charge_2ndpass = image_2.copy()
        # Now we overwrite the charges of all non-core pixels in the camera
        # plus all those pixels which didn't survive the preliminary
        # cleaning.
        # We apply also their corrections.
        charge_2ndpass[non_core_pixels_mask] = charge_no_core

        # Same approach for the pulse times
        pulse_time_2ndpass = pulse_time_1stpass  # core + non-core pixels
        pulse_time_2ndpass[
            non_core_pixels_mask
        ] = pulse_times_no_core  # non-core pixels

        return charge_2ndpass, pulse_time_2ndpass

    def __call__(self, waveforms, telid, selected_gain_channel):
        """
        Call this ImageExtractor.

        Parameters
        ----------
        waveforms : array of shape (N_pixels, N_samples)
            DL0-level waveforms of one event.
        telid : int
            Index of the telescope.
        selected_gain_channel: array of shape (N_channels, N_pixels)
            Array containing the index of the selected gain channel for each
            pixel (0 for low gain, 1 for high gain).

        Returns
        -------
        charge : array_like
            Integrated charge per pixel.
            Shape: (n_pix)
        pulse_time : array_like
            Samples in which the waveform peak has been recognized.
            Shape: (n_pix)
        """

        charge1, pulse_time1, correction1 = self._apply_first_pass(waveforms, telid)

        # FIXME: properly make sure that output is 32Bit instead of downcasting here
        if self.disable_second_pass:
            return (
                (charge1 * correction1[selected_gain_channel]).astype("float32"),
                pulse_time1.astype("float32"),
            )

        charge2, pulse_time2 = self._apply_second_pass(
            waveforms, telid, selected_gain_channel, charge1, pulse_time1, correction1
        )
        # FIXME: properly make sure that output is 32Bit instead of downcasting here
        return charge2.astype("float32"), pulse_time2.astype("float32")
