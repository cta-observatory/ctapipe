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
    'extract_charge_from_peakpos_array',
    'neighbor_average_waveform',
    'extract_pulse_time_weighted_average',
    'subtract_baseline',
]


from abc import abstractmethod
import numpy as np
from traitlets import Int
from ctapipe.core import Component
from numba import njit, prange, float64, float32, int64


def extract_charge_from_peakpos_array(waveforms, peakpos, width, shift):
    """
    Sum the samples from the waveform using the window defined by a
    peak postion, window width, and window shift.

    Parameters
    ----------
    waveforms : ndarray
        Waveforms stored in a numpy array.
        Shape: (n_chan, n_pix, n_samples)
    peakpos : ndarray
        Numpy array of the peak position for each pixel.
        Shape: (n_chan, n_pix)
    width : ndarray or int
        Window size of integration window.
        Shape (if numpy array): (n_chan, n_pix)
    shift : ndarray or int
        Window size of integration window.
        Shape (if numpy array): (n_chan, n_pix)

    Returns
    -------
    charge : ndarray
        Extracted charge.
        Shape: (n_chan, n_pix)
    integration_window : ndarray
        Boolean array indicating which samples were included in the
        charge extraction
        Shape: (n_chan, n_pix, n_samples)

    """
    start = peakpos - shift
    end = start + width
    ind = np.indices(waveforms.shape)[2]
    integration_window = ((ind >= start[..., np.newaxis]) &
                          (ind < end[..., np.newaxis]))
    charge = (waveforms * integration_window).sum(axis=2)

    return charge


@njit([
    float64[:, :, :](float64[:, :, :], int64[:, :], int64),
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


def extract_pulse_time_weighted_average(waveforms):
    """
    Use the weighted average of the waveforms to extract the time of the pulse
    in each pixel

    Parameters
    ----------
    waveforms : ndarray
        Waveforms stored in a numpy array.
        Shape: (n_chan, n_pix, n_samples)

    Returns
    -------
    pulse_time : ndarray
        Floating point pulse time in each pixel
        Shape: (n_chan, n_pix)

    """
    samples_i = np.indices(waveforms.shape)[2]
    pulse_time = np.average(samples_i, weights=waveforms, axis=2)
    outside = np.logical_or(pulse_time < 0, pulse_time >= waveforms.shape[2])
    pulse_time[outside] = -1
    return pulse_time


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
        peakpos : ndarray
            Position of the peak found in each pixel.
            Shape: (n_chan, n_pix)
        window : ndarray
            Bool numpy array defining the samples included in the integration
            window.
            Shape: (n_chan, n_pix, n_samples)
        """


class FullWaveformSum(ImageExtractor):
    """
    Extractor that sums the entire waveform.
    """

    def __call__(self, waveforms):
        charge = waveforms.sum(2)
        pulse_time = extract_pulse_time_weighted_average(waveforms)
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
        pulse_time = extract_pulse_time_weighted_average(waveforms)
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
                'from the peakpos (peakpos - shift)'
    ).tag(config=True)

    def __call__(self, waveforms):
        peakpos = waveforms.mean(1).argmax(1)
        start = peakpos - self.window_shift
        end = start + self.window_width
        charge = np.stack([
            waveforms[0, :, start[0]:end[0]].sum(1),  # HI channel
            waveforms[1, :, start[1]:end[1]].sum(1),  # LO channel
        ])
        pulse_time = extract_pulse_time_weighted_average(waveforms)
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
                'from the peakpos (peakpos - shift)'
    ).tag(config=True)

    def __call__(self, waveforms):
        peakpos = waveforms.argmax(2).astype(np.int)
        charge = extract_charge_from_peakpos_array(
            waveforms, peakpos, self.window_width, self.window_shift
        )
        pulse_time = extract_pulse_time_weighted_average(waveforms)
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
                'from the peakpos (peakpos - shift)'
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
        peakpos = average_wfs.argmax(2)
        charge = extract_charge_from_peakpos_array(
            waveforms, peakpos, self.window_width, self.window_shift
        )
        pulse_time = extract_pulse_time_weighted_average(waveforms)
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
