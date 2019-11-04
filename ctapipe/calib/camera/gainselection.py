"""
Algorithms to select correct gain channel
"""
from abc import abstractmethod
from enum import IntEnum
import numpy as np
from ctapipe.core import Component, traits

__all__ = [
    'GainChannel',
    'GainSelector',
    'ManualGainSelector',
    'ThresholdGainSelector',
]


class GainChannel(IntEnum):
    """
    Possible gain channels
    """
    HIGH = 0
    LOW = 1


class GainSelector(Component):
    """
    Base class for algorithms that decide on the gain channel to use
    """

    def __call__(self, waveforms):
        """
        Reduce the waveform to a single gain channel

        Parameters
        ----------
        waveforms : ndarray
            Waveforms stored in a numpy array of shape
            (n_chan, n_pix, n_samples).

        Returns
        -------
        selected_gain_channel : ndarray
            Gain channel to use for each pixel
            Shape: n_pix
            Dtype: int8
        """
        if waveforms.ndim == 2:  # Return None if already gain selected
            return None
        elif waveforms.ndim == 3:
            n_channels, n_pixels, _ = waveforms.shape
            if n_channels == 1:  # Must be first channel if only one channel
                return np.zeros(n_pixels, dtype=np.int8)
            else:
                return self.select_channel(waveforms)
        else:
            raise ValueError(
                f"Cannot handle waveform array of shape: {waveforms.ndim}"
            )

    @abstractmethod
    def select_channel(self, waveforms):
        """
        Abstract method to be defined by a GainSelector subclass.

        Call the relevant functions to decide on the gain channel used for
        each pixel.

        Parameters
        ----------
        waveforms : ndarray
            Waveforms stored in a numpy array of shape
            (n_chan, n_pix, n_samples).

        Returns
        -------
        selected_gain_channel : ndarray
            Gain channel to use for each pixel
            Shape: n_pix
            Dtype: int8
        """


class ManualGainSelector(GainSelector):
    """
    Manually choose a gain channel.
    """
    channel = traits.CaselessStrEnum(
        ["HIGH", "LOW"],
        default_value="HIGH",
        help="Which gain channel to retain"
    ).tag(config=True)

    def select_channel(self, waveforms):
        n_pixels = waveforms.shape[1]
        return np.full(n_pixels, GainChannel[self.channel])


class ThresholdGainSelector(GainSelector):
    """
    Select gain channel according to a maximum threshold value.
    """
    threshold = traits.Float(
        default_value=4000,
        help="Threshold value in waveform sample units. If a waveform "
             "contains a sample above this threshold, use the low gain "
             "channel for that pixel."
    ).tag(config=True)

    def select_channel(self, waveforms):
        return (waveforms[0] > self.threshold).any(axis=1).astype(np.int8)
