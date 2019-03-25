"""
Charge extraction algorithms to reduce the image to one value per pixel
"""

__all__ = [
    'WaveformReducer',
    'FullWaveformSum',
    'UserWindowSum',
    'GlobalWindowSum',
    'LocalWindowSum',
    'NeighbourWindowSum',
    'extract_charge_from_peakpos_array',
    'extract_pulse_time_weighted_average',
]


from abc import abstractmethod
import numpy as np
from traitlets import Int
from ctapipe.core import Component
from ctapipe.utils.neighbour_sum import get_sum_array


def extract_charge_from_peakpos_array(waveforms, peakpos, width, shift):
    """
    Build the numpy array of bools defining the integration window.

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

    """
    start = peakpos - shift
    end = start + width
    ind = np.indices(waveforms.shape)[2]
    integration_window = (ind >= start[..., None]) & (ind < end[..., None])
    windowed = np.ma.array(waveforms, mask=~integration_window)
    charge = windowed.sum(2).data

    return charge


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
    return pulse_time


class WaveformReducer(Component):

    def __init__(self, config=None, parent=None, **kwargs):
        """
        Base component to handle the extraction of charge and pulse time
        from an image cube.

        Attributes
        ----------
        neighbours : ndarray
            2D array where each row is [pixel index, one neighbour
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

        self.neighbours = None

    @staticmethod
    def requires_neighbours():
        """
        Method used for callers of the WaveformReducer to know if the extractor
        requires knowledge of the pixel neighbours

        Returns
        -------
        bool
        """
        return False

    def check_neighbour_set(self):
        """
        Check if the pixel neighbours has been set for the reducer

        Raises
        -------
        ValueError
            If neighbours has not been set
        """
        if self.requires_neighbours():
            if self.neighbours is None:
                self.log.exception("neighbours attribute must be set")
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


class FullWaveformSum(WaveformReducer):
    """
    Waveform reducer that integrates the entire waveform.
    """

    def __call__(self, waveforms):
        charge = waveforms.sum(2)
        pulse_time = extract_pulse_time_weighted_average(waveforms)
        return charge, pulse_time


class UserWindowSum(WaveformReducer):
    """
    Waveform reducer that integrates within a window defined by the user.
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


class GlobalWindowSum(WaveformReducer):
    """
    Waveform reducer that defines an integration window defined by the
    average waveform across all pixels.
    """
    window_shift = Int(
        3, help='Define the shift of the integration window '
                'from the peakpos (peakpos - shift)'
    ).tag(config=True)
    window_width = Int(
        7, help='Define the width of the integration window'
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


class LocalWindowSum(WaveformReducer):
    """
    Waveform reducer that defines an integration window about the local
    peak in each pixel.
    """
    window_shift = Int(
        3, help='Define the shift of the integration window '
                'from the peakpos (peakpos - shift)'
    ).tag(config=True)
    window_width = Int(
        7, help='Define the width of the integration window'
    ).tag(config=True)

    def __call__(self, waveforms):
        peakpos = waveforms.argmax(2).astype(np.int)
        charge, window = extract_charge_from_peakpos_array(
            waveforms, peakpos, self.window_width, self.window_shift
        )
        pulse_time = extract_pulse_time_weighted_average(waveforms)
        return charge, pulse_time


class NeighbourWindowSum(WaveformReducer):
    """
    Waveform reducer that defines an integration window defined by the
    peaks in the neighbouring pixels.
    """
    window_shift = Int(
        3, help='Define the shift of the integration window '
                'from the peakpos (peakpos - shift)'
    ).tag(config=True)
    window_width = Int(
        7, help='Define the width of the integration window'
    ).tag(config=True)
    lwt = Int(
        0, help='Weight of the local pixel (0: peak from neighbours only, '
                '1: local pixel counts as much as any neighbour)'
    ).tag(config=True)

    def requires_neighbours(self):
        return True

    def __call__(self, waveforms):
        shape = waveforms.shape
        waveforms_32 = waveforms.astype(np.float32)
        sum_data = np.zeros_like(waveforms_32)
        n = self.neighbours.astype(np.uint16)
        get_sum_array(waveforms_32, sum_data, *shape, n, n.shape[0], self.lwt)
        peakpos = sum_data.argmax(2).astype(np.int)
        charge, window = extract_charge_from_peakpos_array(
            waveforms, peakpos, self.window_width, self.window_shift
        )
        pulse_time = extract_pulse_time_weighted_average(waveforms)
        return charge, pulse_time
