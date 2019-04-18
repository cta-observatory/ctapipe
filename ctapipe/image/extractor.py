"""
Charge extraction algorithms to reduce the image to one value per pixel
"""

__all__ = [
    'ImageExtractor',
    'FullWaveformSum',
    'FixedWindowSum',
    'GlobalPeakWindowSum',
    'LocalPeakWindowSum',
    'NeighborPeakWindowSum',
    'BaselineSubtractedNeighborPeakWindowSum',
    'sum_samples_around_peak',
    'neighbor_average_waveform',
    'extract_pulse_time_around_peak',
    'subtract_baseline',
]


from abc import abstractmethod
import numpy as np
from traitlets import Int
from ctapipe.core import Component
from numba import njit, prange, guvectorize, float64, float32, int64, int32


@guvectorize(
    [
        (float64[:], int64, int64, int64, float64[:]),
        (float32[:], int64, int64, int64, float64[:]),
    ],
    '(s),(),(),()->()',
    nopython=True,
)
def sum_samples_around_peak(waveforms, peak_index, width, shift, ret):
    """
    Sum the samples from the waveform using the window defined by a
    peak position, window width, and window shift.

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
        Shape: (n_chan, n_pix, n_samples)
    peak_index : ndarray or int
        Peak index for each pixel.
    width : ndarray or int
        Window size of integration window for each pixel.
    shift : ndarray or int
        Window size of integration window for each pixel.
    ret : ndarray
        Return argument for ufunc (ignore)

    Returns
    -------
    charge : ndarray
        Extracted charge.
        Shape: (n_chan, n_pix)

    """
    n_samples = waveforms.size
    start = peak_index - shift
    end = start + width
    ret[0] = 0
    for sample in prange(start, end):
        if 0 <= sample < n_samples:
            ret[0] += waveforms[sample]


@njit([
    float64[:, :, :](float64[:, :, :], int64[:, :], int64),
    float64[:, :, :](float64[:, :, :], int32[:, :], int64),
    float64[:, :, :](float32[:, :, :], int32[:, :], int64),
    float64[:, :, :](float32[:, :, :], int64[:, :], int64),
], parallel=True)
def neighbor_average_waveform(waveforms, neighbors, lwt):
    """
    Obtain the average waveform built from the neighbors of each pixel

    Parameters
    ----------
    waveforms : ndarray
        Waveforms stored in a numpy array.
        Shape: (n_chan, n_pix, n_samples)
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
        Shape: (n_chan, n_pix, n_samples)

    """
    n_neighbors = neighbors.shape[0]
    sum_ = waveforms * lwt
    n = np.zeros(waveforms.shape)
    for i in prange(n_neighbors):
        pixel = neighbors[i, 0]
        neighbor = neighbors[i, 1]
        for channel in range(waveforms.shape[0]):
            sum_[channel, pixel] += waveforms[channel, neighbor]
            n[channel, pixel] += 1
    return sum_ / n


@guvectorize(
    [
        (float64[:], int64, int64, int64, float64[:]),
        (float32[:], int64, int64, int64, float64[:]),
    ],
    '(s),(),(),()->()',
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
        Shape: (n_chan, n_pix, n_samples)
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
        Shape: (n_chan, n_pix)

    """
    n_samples = waveforms.size
    start = peak_index - shift
    end = start + width

    num = 0
    den = 0
    for isample in prange(start, end):
        if 0 <= isample < n_samples:
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
        Shape: (n_chan, n_pix, n_samples)
    baseline_start : int
        Sample where the baseline window starts
    baseline_end : int
        Sample where the baseline window ends

    Returns
    -------
    baseline_corrected : ndarray
        Waveform with the baseline subtracted
    """
    baseline_corrected = waveforms - np.mean(
        waveforms[..., baseline_start:baseline_end], axis=2
    )[..., None]

    return baseline_corrected


class ImageExtractor(Component):

    def __init__(self, config=None, parent=None, **kwargs):
        """
        Base component to handle the extraction of charge and pulse time
        from an image cube (waveforms).

        Attributes
        ----------
        neighbors : ndarray
            2D array where each row is [pixel index, one neighbor
            of that pixel].
            Changes per telescope.
            Can be obtained from
            `ctapipe.instrument.CameraGeometry.neighbor_matrix_where`.

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
        kwargs
        """
        super().__init__(config=config, parent=parent, **kwargs)

        self.neighbors = None

    @staticmethod
    def requires_neighbors():
        """
        Method used for callers of the ImageExtractor to know if the
        extractor requires knowledge of the pixel neighbors

        Returns
        -------
        bool
        """
        return False

    def check_neighbor_set(self):
        """
        Check if the pixel neighbors has been set for the extractor

        Raises
        -------
        ValueError
            If neighbors has not been set
        """
        if self.requires_neighbors():
            if self.neighbors is None:
                self.log.exception("neighbors attribute must be set")
                raise ValueError()

    @abstractmethod
    def __call__(self, waveforms):
        """
        Call the relevant functions to fully extract the charge and time
        for the particular extractor.

        Parameters
        ----------
        waveforms : ndarray
            Waveforms stored in a numpy array of shape
            (n_chan, n_pix, n_samples).

        Returns
        -------
        charge : ndarray
            Extracted charge.
            Shape: (n_chan, n_pix)
        pulse_time : ndarray
            Floating point pulse time in each pixel.
            Shape: (n_chan, n_pix)
        """


class FullWaveformSum(ImageExtractor):
    """
    Extractor that sums the entire waveform.
    """

    def __call__(self, waveforms):
        charge = waveforms.sum(2)
        pulse_time = extract_pulse_time_around_peak(
            waveforms, 0, waveforms.shape[-1], 0
        )
        return charge, pulse_time


class FixedWindowSum(ImageExtractor):
    """
    Extractor that sums within a fixed window defined by the user.
    """
    window_start = Int(
        0, help='Define the start position for the integration window'
    ).tag(config=True)
    window_width = Int(
        7, help='Define the width of the integration window'
    ).tag(config=True)

    def __call__(self, waveforms):
        start = self.window_start
        end = self.window_start + self.window_width
        charge = waveforms[..., start:end].sum(2)
        pulse_time = extract_pulse_time_around_peak(
            waveforms, self.window_start, self.window_width, 0
        )
        return charge, pulse_time


class GlobalPeakWindowSum(ImageExtractor):
    """
    Extractor which sums in a window about the
    peak from the global average waveform.
    """
    window_width = Int(
        7, help='Define the width of the integration window'
    ).tag(config=True)
    window_shift = Int(
        3, help='Define the shift of the integration window '
                'from the peak_index (peak_index - shift)'
    ).tag(config=True)

    def __call__(self, waveforms):
        peak_index = waveforms.mean(1).argmax(1)
        charge = sum_samples_around_peak(
            waveforms, peak_index[:, np.newaxis],
            self.window_width, self.window_shift
        )
        pulse_time = extract_pulse_time_around_peak(
            waveforms, peak_index[:, np.newaxis],
            self.window_width, self.window_shift
        )
        return charge, pulse_time


class LocalPeakWindowSum(ImageExtractor):
    """
    Extractor which sums in a window about the
    peak in each pixel's waveform.
    """
    window_width = Int(
        7, help='Define the width of the integration window'
    ).tag(config=True)
    window_shift = Int(
        3, help='Define the shift of the integration window '
                'from the peak_index (peak_index - shift)'
    ).tag(config=True)

    def __call__(self, waveforms):
        peak_index = waveforms.argmax(2).astype(np.int)
        charge = sum_samples_around_peak(
            waveforms, peak_index, self.window_width, self.window_shift
        )
        pulse_time = extract_pulse_time_around_peak(
            waveforms, peak_index, self.window_width, self.window_shift
        )
        return charge, pulse_time


class NeighborPeakWindowSum(ImageExtractor):
    """
    Extractor which sums in a window about the
    peak defined by the wavefroms in neighboring pixels.
    """
    window_width = Int(
        7, help='Define the width of the integration window'
    ).tag(config=True)
    window_shift = Int(
        3, help='Define the shift of the integration window '
                'from the peak_index (peak_index - shift)'
    ).tag(config=True)
    lwt = Int(
        0, help='Weight of the local pixel (0: peak from neighbors only, '
                '1: local pixel counts as much as any neighbor)'
    ).tag(config=True)

    def requires_neighbors(self):
        return True

    def __call__(self, waveforms):
        average_wfs = neighbor_average_waveform(
            waveforms, self.neighbors, self.lwt
        )
        peak_index = average_wfs.argmax(2)
        charge = sum_samples_around_peak(
            waveforms, peak_index, self.window_width, self.window_shift
        )
        pulse_time = extract_pulse_time_around_peak(
            waveforms, peak_index, self.window_width, self.window_shift
        )
        return charge, pulse_time


class BaselineSubtractedNeighborPeakWindowSum(NeighborPeakWindowSum):
    """
    Extractor that first subtracts the baseline before summing in a
    window about the peak defined by the wavefroms in neighboring pixels.
    """
    baseline_start = Int(
        0, help='Start sample for baseline estimation'
    ).tag(config=True)
    baseline_end = Int(
        10, help='End sample for baseline estimation'
    ).tag(config=True)

    def __call__(self, waveforms):
        baseline_corrected = subtract_baseline(
            waveforms, self.baseline_start, self.baseline_end
        )
        return super().__call__(baseline_corrected)
