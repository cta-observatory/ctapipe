"""
Charge extraction algorithms to reduce the image to one value per pixel
"""

__all__ = [
    'ChargeExtractor',
    'FullIntegrator',
    'SimpleIntegrator',
    'GlobalPeakIntegrator',
    'LocalPeakIntegrator',
    'NeighbourPeakIntegrator',
    'AverageWfPeakIntegrator',
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
    # Ensure window is within waveform
    # width[width > n_samples] = n_samples
    # start[start < 0] = 0
    # sum_check = start + width > n_samples
    # start[sum_check] = n_samples - width[sum_check]

    start = peakpos - shift
    end = start + width
    ind = np.indices(waveforms.shape)[2]
    integration_window = (ind >= start[..., None]) & (ind < end[..., None])
    windowed = np.ma.array(waveforms, mask=~integration_window)
    charge = windowed.sum(2).data

    # TODO: remove integration window return
    return charge, integration_window


class ChargeExtractor(Component):

    def __init__(self, config=None, parent=None, **kwargs):
        """
        Base component to handle the extraction of charge from an image cube.

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
        Method used for callers of the ChargeExtractor to know if the extractor
        requires knowledge of the pixel neighbours

        Returns
        -------
        bool
        """
        return False

    def check_neighbour_set(self):
        """
        Check if the pixel neighbours has been set for the extractor

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
    def extract_charge(self, waveforms):
        """
        Call the relevant functions to fully extract the charge for the
        particular extractor.

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


class FullIntegrator(ChargeExtractor):
    """
    Charge extractor that integrates the entire waveform.
    """

    def extract_charge(self, waveforms):
        # TODO: remove integration window return
        peakpos = np.zeros(waveforms.shape[:2], dtype=np.intp)
        window = np.ones(waveforms.shape, dtype=np.bool)
        return waveforms.sum(2), peakpos, window


class SimpleIntegrator(ChargeExtractor):
    """
    Charge extractor that integrates within a window defined by the user.
    """
    window_start = Int(
        0, help='Define the start position for the integration window'
    ).tag(config=True)
    window_width = Int(
        7, help='Define the width of the integration window'
    ).tag(config=True)

    def extract_charge(self, waveforms):
        start = self.window_start
        end = self.window_start + self.window_width
        # TODO: remove integration window return
        peakpos = np.zeros(waveforms.shape[:2], dtype=np.intp)
        window = np.ones(waveforms.shape, dtype=np.bool)
        return waveforms[..., start:end].sum(2), peakpos, window


class GlobalPeakIntegrator(ChargeExtractor):
    """
    Charge extractor that defines an integration window about the global
    peak in the image.
    """
    window_shift = Int(
        3, help='Define the shift of the integration window '
                'from the peakpos (peakpos - shift)'
    ).tag(config=True)
    window_width = Int(
        7, help='Define the width of the integration window'
    ).tag(config=True)

    def extract_charge(self, waveforms):
        max_t = waveforms.argmax(2)
        max_s = waveforms.max(2)
        peakpos = np.round(
            np.average(max_t, weights=max_s, axis=1)
        ).astype(np.int)
        start = peakpos - self.window_shift
        end = start + self.window_width
        charge = np.stack([
            waveforms[0, :, start[0]:end[0]].sum(1),  # HI channel
            waveforms[1, :, start[1]:end[1]].sum(1),  # LO channel
        ])

        # TODO: remove integration window return
        ind = np.indices(waveforms.shape)[2]
        window = (ind >= start[..., None, None]) & (ind < end[..., None, None])

        return charge, peakpos, window


class LocalPeakIntegrator(ChargeExtractor):
    """
    Charge extractor that defines an integration window about the local
    peak in each pixel.
    """
    window_shift = Int(
        3, help='Define the shift of the integration window '
                'from the peakpos (peakpos - shift)'
    ).tag(config=True)
    window_width = Int(
        7, help='Define the width of the integration window'
    ).tag(config=True)

    def extract_charge(self, waveforms):
        peakpos = waveforms.argmax(2).astype(np.int)
        charge, window = extract_charge_from_peakpos_array(
            waveforms, peakpos, self.window_width, self.window_shift
        )
        return charge, peakpos, window


class NeighbourPeakIntegrator(ChargeExtractor):
    """
    Charge extractor that defines an integration window defined by the
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

    def extract_charge(self, waveforms):
        shape = waveforms.shape
        waveforms_32 = waveforms.astype(np.float32)
        sum_data = np.zeros_like(waveforms_32)
        n = self.neighbours.astype(np.uint16)
        get_sum_array(waveforms_32, sum_data, *shape, n, n.shape[0], self.lwt)
        peakpos = sum_data.argmax(2).astype(np.int)
        charge, window = extract_charge_from_peakpos_array(
            waveforms, peakpos, self.window_width, self.window_shift
        )
        return charge, peakpos, window


class AverageWfPeakIntegrator(ChargeExtractor):
    """
    Charge extractor that defines an integration window defined by the
    average waveform across all pixels.
    """
    window_shift = Int(
        3, help='Define the shift of the integration window '
                'from the peakpos (peakpos - shift)'
    ).tag(config=True)
    window_width = Int(
        7, help='Define the width of the integration window'
    ).tag(config=True)

    def extract_charge(self, waveforms):
        peakpos = waveforms.mean(1).argmax(1)
        start = peakpos - self.window_shift
        end = start + self.window_width
        charge = np.stack([
            waveforms[0, :, start[0]:end[0]].sum(1),  # HI channel
            waveforms[1, :, start[1]:end[1]].sum(1),  # LO channel
        ])

        # TODO: remove integration window return
        ind = np.indices(waveforms.shape)[2]
        window = (ind >= start[..., None, None]) & (ind < end[..., None, None])

        return charge, peakpos, window
