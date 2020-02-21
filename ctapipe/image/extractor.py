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
]


from abc import abstractmethod
import numpy as np
from traitlets import Int
from ctapipe.core.traits import IntTelescopeParameter
from ctapipe.core import Component
from numba import njit, prange, guvectorize, float64, float32, int64

from ctapipe.image.cleaning import number_of_islands, largest_island, tailcuts_clean
from ctapipe.image.timing_parameters import timing_parameters
from ctapipe.image.hillas import hillas_parameters, camera_to_shower_coordinates


@guvectorize(
    [
        (float64[:], int64, int64, int64, float64[:], float64[:]),
        (float32[:], int64, int64, int64, float64[:], float64[:]),
    ],
    "(s),(),(),()->(),()",
    nopython=True,
)
def extract_around_peak(waveforms, peak_index, width, shift, sum_, pulse_time):
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
    sum_ : ndarray
        Return argument for ufunc (ignore)
        Returns the sum
    pulse_time : ndarray
        Return argument for ufunc (ignore)
        Returns the pulse_time

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

    # TODO: Return pulse time in units of ns instead of isample
    pulse_time[0] = time_num / time_den if time_den > 0 else peak_index


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
    n = np.zeros(waveforms.shape, dtype=np.int32)
    for i in prange(n_neighbors):
        pixel = neighbors[i, 0]
        neighbor = neighbors[i, 1]
        sum_[pixel] += waveforms[neighbor]
        n[pixel] += 1
    return sum_ / n


@guvectorize(
    [
        (float64[:], int64, int64, int64, float64[:]),
        (float32[:], int64, int64, int64, float64[:]),
    ],
    "(s),(),(),()->()",
    nopython=True,
)
def extract_pulse_time_around_peak(waveforms, peak_index, width, shift, ret):
    """
    Obtain the pulse time within a window defined by a peak finding algorithm,
    using the weighted average of the samples.

    This function is a numpy universal function which defines the operation
    applied on the waveform for every channel and pixel. Therefore in the
    code body of this function:
        - waveforms is a 1D array of size n_samples.
        - Peakpos, width and shift are integers, corresponding to the correct
            value for the current pixel

    The ret argument is required by numpy to create the numpy array which is
    returned. It can be ignored when calling this function.

    Parameters
    ----------
    waveforms : ndarray
        Waveforms stored in a numpy array.
        Shape: (n_pix, n_samples)
    peak_index : ndarray or int
        Peak index in waveform for each pixel.
    width : ndarray or int
        Window size of integration window for each pixel.
    shift : ndarray or int
        Window size of integration window for each pixel.
    ret : ndarray
        Return argument for ufunc (ignore)

    Returns
    -------
    pulse_time : ndarray
        Floating point pulse time in each pixel
        Shape: (n_pix)

    """
    n_samples = waveforms.size
    start = peak_index - shift
    end = start + width

    num = 0
    den = 0
    for isample in prange(start, end):
        if (0 <= isample < n_samples) & (waveforms[isample] > 0):
            num += waveforms[isample] * isample
            den += waveforms[isample]

    # TODO: Return pulse time in units of ns instead of isample
    ret[0] = num / den if den > 0 else peak_index


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


class ImageExtractor(Component):
    def __init__(self, config=None, parent=None, subarray=None, **kwargs):
        """
        Base component to handle the extraction of charge and pulse time
        from an image cube (waveforms).

        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool : ctapipe.core.Tool or None
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        subarray: ctapipe.instrument.SubarrayDescription
            Description of the subarray
        kwargs
        """
        super().__init__(config=config, parent=parent, **kwargs)
        self.subarray = subarray
        for trait in list(self.class_traits()):
            try:
                getattr(self, trait).attach_subarray(subarray)
            except (AttributeError, TypeError):
                pass

    @abstractmethod
    def __call__(self, waveforms, telid=None):
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
            If None, the subarray global default value is used

        Returns
        -------
        charge : ndarray
            Extracted charge.
            Shape: (n_pix)
        pulse_time : ndarray
            Floating point pulse time in each pixel.
            Shape: (n_pix)
        """


class FullWaveformSum(ImageExtractor):
    """
    Extractor that sums the entire waveform.
    """

    def __call__(self, waveforms, telid=None):
        charge, pulse_time = extract_around_peak(waveforms, 0, waveforms.shape[-1], 0)
        return charge, pulse_time


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

    def __call__(self, waveforms, telid=None):
        charge, pulse_time = extract_around_peak(
            waveforms, self.window_start[telid], self.window_width[telid], 0
        )
        return charge, pulse_time


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

    def __call__(self, waveforms, telid=None):
        peak_index = waveforms.mean(axis=-2).argmax(axis=-1)
        charge, pulse_time = extract_around_peak(
            waveforms, peak_index, self.window_width[telid], self.window_shift[telid]
        )
        return charge, pulse_time


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

    def __call__(self, waveforms, telid=None):
        peak_index = waveforms.argmax(axis=-1).astype(np.int)
        charge, pulse_time = extract_around_peak(
            waveforms, peak_index, self.window_width[telid], self.window_shift[telid]
        )
        return charge, pulse_time


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

    def __call__(self, waveforms, telid=None):
        neighbors = self.subarray.tel[telid].camera.neighbor_matrix_where
        average_wfs = neighbor_average_waveform(waveforms, neighbors, self.lwt[telid])
        peak_index = average_wfs.argmax(axis=-1)
        charge, pulse_time = extract_around_peak(
            waveforms, peak_index, self.window_width[telid], self.window_shift[telid]
        )
        return charge, pulse_time


class BaselineSubtractedNeighborPeakWindowSum(NeighborPeakWindowSum):
    """
    Extractor that first subtracts the baseline before summing in a
    window about the peak defined by the wavefroms in neighboring pixels.
    """

    baseline_start = Int(0, help="Start sample for baseline estimation").tag(
        config=True
    )
    baseline_end = Int(10, help="End sample for baseline estimation").tag(config=True)

    def __call__(self, waveforms, telid=None):
        baseline_corrected = subtract_baseline(
            waveforms, self.baseline_start, self.baseline_end
        )
        return super().__call__(baseline_corrected, telid)


class TwoPassWindowSum(ImageExtractor):
    """Extractor which integrates the waveform a second time using a
    time-gradient linear fit. This is in particular the CTA-MARS version.

    Procedure:
    1) slide a 3-samples window through the waveform, finding max counts sum;
       the range of the sliding is the one allowing extension from 3 to 5;
       add 1 sample on each side and integrate charge in the 5-sample window;
       time is obtained as a charge-weighted average of the sample numbers;
       No information from neighboouring pixels is used.
    2) Preliminary image cleaning via simple tailcut with minimum number
       of core neighbours set at 1,
    3) Only the biggest cluster of pixels is kept.
    4) Parametrize following Hillas approach only if the resulting image has 3
       or more pixels.
    5) Do a linear fit of pulse time vs. distance along major image axis
       (CTA-MARS uses ROOT "robust" fit option,
       aka Least Trimmed Squares, to get rid of far outliers - this should
       be implemented in 'timing_parameters', e.g scipy.stats.siegelslopes).
    6) For all pixels except the core ones in the main island, integrate
       the waveform once more, in a fixed window of 5 samples set at the time
       "predicted" by the linear time fit.
       If the predicted time for a pixel leads to a window outside the readout
       window, then integrate the last (or first) 5 samples.
    7) The result is an image with main-island core pixels calibrated with a
       1st pass and non-core pixels re-calibrated with a 2nd pass.

    """

    def __call__(self, waveforms, telid=None):
        """
        Call this ImageExtractor.

        Parameters
        ----------
        waveforms : array of size (N_pixels, N_samples)
            DL0-level waveforms of one event.

        Returns
        -------
        charge : array_like
            Integrated charge per pixel.
            Shape: (n_pix, n_channels)
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
            waveforms, startWindows + 1, window_widths, window_shifts
        )

        # STEP 2

        # Set thresholds for core-pixels depending on telescope type.
        # Boundary thresholds will be half of core thresholds.
        # WARNING: these values should be read from a configuration file
        # à-la-protopipe and they will depend in principle also on camera type.
        subarray = self.subarray
        if subarray.tel[telid].type == "LST":
            core_th = 6  # (not yet optimized)
        if subarray.tel[telid].type == "MST":
            core_th = 8  # (not yet optimized)
        if subarray.tel[telid].type == "SST":
            core_th = 4  # (not yet optimized)

        # Preliminary image cleaning with simple two-level tail-cut
        camera = self.subarray.tel[telid].camera
        mask_1 = tailcuts_clean(
            camera,
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
        num_islands, labels = number_of_islands(camera, mask_1)
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
        if np.count_nonzero(image_2) < 3:
            # we return the 1st pass information
            return charge_1stpass, pulse_time_1stpass
        else:  # otherwise we proceed by parametrizing the image
            hillas = hillas_parameters(camera, image_2)
            # at the end there should be a "quality label" variable to record
            # if a image had less then 3 pixels OR more than three but
            # patologically placed such that width gets parametrized either as
            # 0 or NaN
            # This is what I do in protopipe.

            # STEP 5

            # linear fit of pulse time vs. distance along major image axis
            timing = timing_parameters(camera, image_2, pulse_time_1stpass, hillas)

            long, trans = camera_to_shower_coordinates(
                camera.pix_x, camera.pix_y, hillas.x, hillas.y, hillas.psi
            )

            # WARNING: for LSTCam and NectarCam sample = ns, but not for other
            # cameras
            # Here each pulse time is treated as sample,
            # but a general method should scale it depending on each cam_id
            predicted_pulse_times = (
                timing.slope * long[nonCore_pixels_ids] + timing.intercept
            )
            predicted_peaks = np.zeros(len(predicted_pulse_times))

            # Approximate to nearest integer then cast to int64
            # otherwise 'extract_around_peak' complains
            np.rint(predicted_pulse_times.value, predicted_peaks)
            predicted_peaks = predicted_peaks.astype(np.int64)

            # Due to the fit these peak indexes can now be also outside of the
            # readout window.

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
                nonCore_waveforms, predicted_peaks, window_widths, window_shifts
            )

            # STEP 7

            # combine core and non-core pixels in the final output
            charge_2npass = image_2.copy()  # core + non-core pixels
            charge_2npass[nonCore_pixels_mask] = charge_noCore  # non-core pixels
            pulse_time_2npass = pulse_time_1stpass  # core + non-core pixels
            pulse_time_2npass[
                nonCore_pixels_mask
            ] = pulse_times_noCore  # non-core pixels

            return charge_2npass, pulse_time_2npass
