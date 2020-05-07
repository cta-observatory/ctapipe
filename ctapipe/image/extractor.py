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
from traitlets import Int
from ctapipe.core.traits import IntTelescopeParameter, FloatTelescopeParameter
from ctapipe.core import TelescopeComponent
from numba import njit, prange, guvectorize, float64, float32, int64

from .cleaning import number_of_islands, largest_island, tailcuts_clean
from .timing_parameters import timing_parameters
from .hillas import hillas_parameters, camera_to_shower_coordinates


@guvectorize(
    [
        (float64[:], int64, int64, int64, float64, float64[:], float64[:]),
        (float32[:], int64, int64, int64, float64, float64[:], float64[:]),
    ],
    "(s),(),(),(),()->(),()",
    nopython=True,
)
def extract_around_peak(
<<<<<<< HEAD
    waveforms, peak_index, width, shift, sampling_rate_ghz, sum_, pulse_time
=======
        waveforms, peak_index, width, shift, sampling_rate_ghz, sum_, peak_time
>>>>>>> master
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
    sum_[0] = 0
    time_num = 0
    time_den = 0
    for isample in prange(start, end):
        if 0 <= isample < n_samples:
            sum_[0] += waveforms[isample]
            if waveforms[isample] > 0:
                time_num += waveforms[isample] * isample
                time_den += waveforms[isample]
    peak_time[0] = time_num / time_den if time_den > 0 else peak_index

    # Convert to units of ns
    peak_time[0] /= sampling_rate_ghz


@njit(parallel=True)
def neighbor_average_waveform(waveforms, neighbors, lwt):
    """
    Obtain the average waveform built from the neighbors of each pixel

    Parameters
    ----------
    waveforms : ndarray
        Waveforms stored in a numpy array.
        Shape: (n_pix, n_samples)
    neighbors : ndarray
        2D array where each row is [pixel index, one neighbor of that pixel].
        Changes per telescope.
        Can be obtained from
        `ctapipe.instrument.CameraGeometry.neighbor_matrix_where`.
    lwt: int
        Weight of the local pixel (0: peak from neighbors only,
        1: local pixel counts as much as any neighbor)

    Returns
    -------
    average_wf : ndarray
        Average of neighbor waveforms for each pixel.
        Shape: (n_pix, n_samples)

    """
    n_neighbors = neighbors.shape[0]
    sum_ = waveforms * lwt
    n = np.full(waveforms.shape, lwt, dtype=np.int32)
    for i in prange(n_neighbors):
        pixel = neighbors[i, 0]
        neighbor = neighbors[i, 1]
        sum_[pixel] += waveforms[neighbor]
        n[pixel] += 1
    return sum_ / n


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


def slide_window(waveform, width):
    """Smooth a pixel's waveform (or a slice of it) with a kernel of certain
     size via convolution.

    Parameters
    ----------
    waveform : array_like
        DL0-level waveform (or slice of it) of one event.
        Shape: max (n_samples)
    width : int
        Size of the smoothing kernel.

    Returns
    -------
    sum : array_like
        Array containing the sums for each of the kernel positions.
        Shape: max (n_samples - (window_width - 1))

    """
    sums = np.convolve(waveform, np.ones(width, dtype=int), "valid")
    return sums


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

    window_start = IntTelescopeParameter(
        default_value=0, help="Define the start position for the integration window"
    ).tag(config=True)
    window_width = IntTelescopeParameter(
        default_value=7, help="Define the width of the integration window"
    ).tag(config=True)

    @lru_cache(maxsize=128)
    def _calculate_correction(self, telid):
        """
        Calculate the correction for the extracted change such that the value
        returned would equal 1 for a noise-less unit pulse.

        Assuming the pulse is centered in the manually defined integration
        window, the integration_correction with a shift=0 is correct.
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
            0,
        )

    def __call__(self, waveforms, telid, selected_gain_channel):
<<<<<<< HEAD
        charge, pulse_time = extract_around_peak(
            waveforms,
            self.window_start.tel[telid],
            self.window_width.tel[telid],
            0,
            self.sampling_rate[telid],
=======
        charge, peak_time = extract_around_peak(
            waveforms, self.window_start.tel[telid], self.window_width.tel[telid], 0,
            self.sampling_rate[telid]
>>>>>>> master
        )
        correction = self._calculate_correction(telid=telid)[selected_gain_channel]
        return charge * correction, peak_time


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
        correction = self._calculate_correction(telid=telid)[selected_gain_channel]
        return charge * correction, peak_time


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
        correction = self._calculate_correction(telid=telid)[selected_gain_channel]
        return charge * correction, peak_time


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
        neighbors = self.subarray.tel[telid].camera.geometry.neighbor_matrix_where
        average_wfs = neighbor_average_waveform(
            waveforms, neighbors, self.lwt.tel[telid]
        )
        peak_index = average_wfs.argmax(axis=-1)
        charge, peak_time = extract_around_peak(
            waveforms,
            peak_index,
            self.window_width.tel[telid],
            self.window_shift.tel[telid],
            self.sampling_rate[telid],
        )
        correction = self._calculate_correction(telid=telid)[selected_gain_channel]
        return charge * correction, peak_time


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

    # Boolean that is used to disable the 2np pass and return the 1st pass
    disable_second_pass = False

    def _calculate_correction(self, telid, widths, shifts, selected_gain_channel):
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
        widths : array of shape N_pixels
            Width of the integration window (in units of n_samples)
        shifts : array of shape N_pixels
            Values of the window shifts per pixel.

        Returns
        -------
        correction : ndarray
            Value of the pixel-wise gain-selected integration correction.

        """
        readout = self.subarray.tel[telid].camera.readout
        # Calculate correction of first pixel for both channels
        correction = integration_correction(
            readout.reference_pulse_shape,
            readout.reference_pulse_sample_width.to_value("ns"),
            (1 / readout.sampling_rate).to_value("ns"),
            widths[0],
            shifts[0],
        )
        # then do the same for each remaining pixel and attach the result as
        # a column containing information from both channels
        for pixel in range(len(selected_gain_channel)):
            new_pixel_both_channels = integration_correction(
                readout.reference_pulse_shape,
                readout.reference_pulse_sample_width.to_value("ns"),
                (1 / readout.sampling_rate).to_value("ns"),
                widths[pixel],
                shifts[pixel],
            )
            # stack the columns (i.e pixels) so the final correction array
            # is N_channels X N_pixels
            correction = np.column_stack((correction, new_pixel_both_channels))

        # select the right channel per pixel
        correction = np.asarray(
            [
                correction[:, pix_id][selected_gain_channel[pix_id]]
                for pix_id in range(len(selected_gain_channel))
            ]
        )
        return correction

    def _apply_first_pass(self, waveforms, telid, selected_gain_channel):
        """
        Execute step 1.

        Parameters
        ----------
        waveforms : array of size (N_pixels, N_samples)
            DL0-level waveforms of one event.
        telid : int
            Index of the telescope.
        selected_gain_channel: array of size (N_channels, N_pixels)
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
        # STEP 1

        # Starting from DL0, the channel is already selected (if more than one)
        # event.dl0.tel[tel_id].waveform object has shape (N_pixels, N_samples)

        # For each pixel, we slide a 3-samples window through the
        # waveform without touching the extremes (so later we can increase it
        # to 5), summing each time the ADC counts contained within it.

        # 'width' could be configurable in a generalized version
        # Right now this image extractor is optimized for LSTCam and NectarCam
        width = 3
        sums = np.apply_along_axis(slide_window, 1, waveforms[:, 1:-1], width)
        # Note that the input waveforms are clipped at the extremes because
        # we want to extend this 3-samples window to 5 samples
        # 'sums' has now the shape of (N_pixels, N_samples-4)

        # For each pixel, in each of the (N_samples - 4) positions, we check
        # where the window encountered the maximum number of ADC counts
        startWindows = np.apply_along_axis(np.argmax, 1, sums)
        # Now startWindows has the shape of (N_pixels).
        # Note that the index values stored in startWindows come from 'sums'
        # of which the first index (0) corresponds of index 1 of each waveform
        # since we clipped them before.

        # Since we have to add 1 sample on each side, window_shift will always
        # be (-)1, while window_width will always be window1_width + 1
        # so we the final 5-samples window will be 1+3+1
        window_widths = np.full_like(startWindows, width + 1)
        window_shifts = np.full_like(startWindows, 1)

        # the 'peak_index' argument of 'extract_around_peak' has a different
        # meaning here: it's the start of the 3-samples window.
        # Since since the "sums" arrays started from index 1 of each waveform,
        # then each peak index has to be increased by one
        charge_1stpass, pulse_time_1stpass = extract_around_peak(
            waveforms,
            startWindows + 1,
            window_widths,
            window_shifts,
            self.sampling_rate[telid],
        )

        # Get integration correction factors
        correction = self._calculate_correction(
            telid, window_widths, window_shifts, selected_gain_channel
        )

        return charge_1stpass, pulse_time_1stpass, correction

    def _apply_second_pass(
        self,
        waveforms,
        telid,
        selected_gain_channel,
        charge_1stpass,
        pulse_time_1stpass,
        correction,
    ):
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
        charge_1stpass : array of shape N_pixels
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
        charge_1stpass = charge_1stpass * correction

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
        nonCore_pixels_ids = np.where(image_2 < core_th)[0]
        nonCore_pixels_mask = image_2 < core_th

        # STEP 4

        # if the resulting image has less then 3 pixels
        # or there are more than 3 pixels but all contain a number of
        # photoelectrons above the core threshold
        if np.count_nonzero(image_2) < 3:
            # we return the 1st pass information
            # NOTE: In this case, the image was not bright enough!
            # We should label it as "bad and NOT use it"
            return charge_1stpass, pulse_time_1stpass
        elif len(nonCore_pixels_ids) == 0:
            # Since all reconstructed charges are above the core threshold,
            # there is no need to perform the 2nd pass.
            # We return the 1st pass information.
            # NOTE: In this case, even if this is 1st pass information,
            # the image is actually very bright! We should label it as "good"!
            return charge_1stpass, pulse_time_1stpass
        else:  # otherwise we proceed by parametrizing the image
            hillas = hillas_parameters(camera_geometry, image_2)

            # STEP 5

            # linear fit of pulse time vs. distance along major image axis
            # using only the main island surviving the preliminary
            # image cleaning
            # WARNING: in case of outliers, the fit can perform better if
            # it is a robust algorithm.
            timing = timing_parameters(
                camera_geometry, image_2, pulse_time_1stpass, hillas
            )

            # get projected distances along main image axis
            long, _ = camera_to_shower_coordinates(
                camera_geometry.pix_x,
                camera_geometry.pix_y,
                hillas.x,
                hillas.y,
                hillas.psi,
            )

            # get the predicted times as a linear relation
            predicted_pulse_times = (
                timing.slope * long[nonCore_pixels_ids] + timing.intercept
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
            nonCore_waveforms = waveforms[nonCore_pixels_ids]

            # Build 'width' and 'shift' arrays that adapt on the position of the
            # window along each waveform

            # Now the definition of peak_index is really the peak.
            # We have to add 2 samples each side, so the shist will always
            # be (-)2, while width will always end 4 samples to the right.
            # This "always" refers to a 5-samples window of course
            window_widths = np.full_like(predicted_peaks, 4, dtype=np.int64)
            window_shifts = np.full_like(predicted_peaks, 2, dtype=np.int64)

            # BUT, if the resulting 5-samples window falls outside of the readout
            # window then we take the first (or last) 5 samples
            window_widths[predicted_peaks < 0] = 4
            window_shifts[predicted_peaks < 0] = 0
            window_widths[predicted_peaks > (waveforms.shape[1] - 1)] = 4
            window_shifts[predicted_peaks > (waveforms.shape[1] - 1)] = 4

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
            charge_noCore, pulse_times_noCore = extract_around_peak(
                nonCore_waveforms,
                predicted_peaks,
                window_widths,
                window_shifts,
                self.sampling_rate[telid],
            )

            # Modify integration correction factors only for non-core pixels
            correction_2ndPass = self._calculate_correction(
                telid,
                window_widths,
                window_shifts,
                selected_gain_channel[nonCore_pixels_ids],
            )
            np.put(correction, [nonCore_pixels_ids], correction_2ndPass)

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
            charge_2ndpass[nonCore_pixels_mask] = charge_noCore * correction_2ndPass

            # Same approach for the pulse times
            pulse_time_2npass = pulse_time_1stpass  # core + non-core pixels
            pulse_time_2npass[
                nonCore_pixels_mask
            ] = pulse_times_noCore  # non-core pixels

            return charge_2ndpass, pulse_time_2npass

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

        charge1, pulse_time1, correction1 = self._apply_first_pass(
            waveforms, telid, selected_gain_channel
        )

        if self.disable_second_pass:
            return charge1 * correction1, pulse_time1
        else:
            charge2, pulse_time2 = self._apply_second_pass(
                waveforms,
                telid,
                selected_gain_channel,
                charge1,
                pulse_time1,
                correction1,
            )
            return charge2, pulse_time2
