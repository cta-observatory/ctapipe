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
    "extract_around_peak",
    "neighbor_average_waveform",
    "subtract_baseline",
    "integration_correction"
]


from abc import abstractmethod
from functools import lru_cache
import numpy as np
from traitlets import Int
from ctapipe.core.traits import IntTelescopeParameter
from ctapipe.core import TelescopeComponent
from numba import njit, prange, guvectorize, float64, float32, int64


@guvectorize(
    [
        (float64[:], int64, int64, int64, float64, float64[:], float64[:]),
        (float32[:], int64, int64, int64, float64, float64[:], float64[:]),
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
    reference_pulse_shape, reference_pulse_sample_width_ns, sample_width_ns,
    window_width, window_shift
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
            telid: telescope.camera.readout.sampling_rate.to_value('GHz')
            for telid, telescope in subarray.tel.items()
        }

    @abstractmethod
    def _calculate_correction(self, telid):
        """
        Calculate the correction for the extracted change such that the value
        returned would equal 1 for a noise-less unit pulse.

        Decorate this method with @lru_cache to ensure it is only calculated
        once per telescope

        Parameters
        ----------
        telid : int

        Returns
        -------
        correction : ndarray
        The correction to apply to an extracted charge using this ImageExtractor
        Has size n_channels, as a different correction value might be required
        for different gain channels
        """

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

    def _calculate_correction(self, telid):
        """
        No correction is required, as the full pulse has been integrated.
        """
        return 1

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
        Assuming the pulse is centered in the manually defined integration
        window, the integration_correction with a shift=0 is correct
        """
        readout = self.subarray.tel[telid].camera.readout
        return integration_correction(
            readout.reference_pulse_shape,
            readout.reference_pulse_sample_width.to_value('ns'),
            (1/readout.sampling_rate).to_value('ns'),
            self.window_width.tel[telid],
            0,
        )

    def __call__(self, waveforms, telid, selected_gain_channel):
        charge, peak_time = extract_around_peak(
            waveforms, self.window_start.tel[telid], self.window_width.tel[telid], 0,
            self.sampling_rate[telid]
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
        readout = self.subarray.tel[telid].camera.readout
        return integration_correction(
            readout.reference_pulse_shape,
            readout.reference_pulse_sample_width.to_value('ns'),
            (1/readout.sampling_rate).to_value('ns'),
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
            self.sampling_rate[telid]
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
        readout = self.subarray.tel[telid].camera.readout
        return integration_correction(
            readout.reference_pulse_shape,
            readout.reference_pulse_sample_width.to_value('ns'),
            (1/readout.sampling_rate).to_value('ns'),
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
            self.sampling_rate[telid]
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
        readout = self.subarray.tel[telid].camera.readout
        return integration_correction(
            readout.reference_pulse_shape,
            readout.reference_pulse_sample_width.to_value('ns'),
            (1/readout.sampling_rate).to_value('ns'),
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
            self.sampling_rate[telid]
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
