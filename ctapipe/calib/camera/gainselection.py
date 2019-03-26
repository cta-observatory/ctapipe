"""
Algorithms to select correct gain channel
"""
from abc import ABCMeta, abstractclassmethod

import numpy as np

from ...core import Component, traits
from ...utils import get_table_dataset

__all__ = ['GainSelector',
           'ThresholdGainSelector',
           'SimpleGainSelector',
           'pick_gain_channel']


def pick_gain_channel(waveforms, threshold, select_by_sample=False):
    """
    the PMTs on some cameras have 2 gain channels. select one
    according to a threshold.

    Parameters:
    -----------
    waveforms: np.ndarray
        Array of shape (N_gain, N_pix, N_samp)
    threshold: float
        threshold (in PE/sample) of when to switch to low-gain channel
    select_by_sample: bool
        if true, select only low-gain *samples* when high-gain is over
        threshold

    Returns
    -------
    tuple:
        gain-selected intensity, boolean array of which channel was chosen
    """

    # if we have 2 channels:
    if waveforms.shape[0] == 2:
        waveforms = np.squeeze(waveforms)
        new_waveforms = waveforms[0].copy()

        if select_by_sample:
            # replace any samples that are above threshold with low-gain ones:
            gain_mask = waveforms[0] > threshold
            new_waveforms[gain_mask] = waveforms[1][gain_mask]
        else:
            # use entire low-gain waveform if any sample of high-gain
            # waveform is above threshold
            gain_mask = (waveforms[0] > threshold).any(axis=1)
            new_waveforms[gain_mask] = waveforms[1][gain_mask]

    elif waveforms.shape[0] == 1:
        new_waveforms = np.squeeze(waveforms)
        gain_mask = np.zeros_like(new_waveforms).astype(bool)

    else:
        raise ValueError("input waveforms has shape %s. not sure what to do "
                         "with that.", waveforms.shape)

    return new_waveforms, gain_mask


class GainSelector(Component, metaclass=ABCMeta):
    """
    Base class for algorithms that reduce a 2-gain-channel waveform to a
    single waveform.
    """
    @abstractclassmethod
    def select_gains(self, cam_id, multi_gain_waveform):
        """
        Takes an input waveform and cam_id  and performs gain selection

        Returns
        -------
        tuple(ndarray, ndarray):
            (waveform, gain_mask), where the gain_mask is a boolean array of
            which gain channel was used.
        """
        pass


class NullGainSelector(GainSelector):
    """
    do no gain selection, leaving possibly 2 gain channels at the DL1 level.
    this may break further steps in the chain if they do not expect 2 gains.
    """

    def select_gains(self, cam_id, multi_gain_waveform):
        return multi_gain_waveform, np.ones(multi_gain_waveform.shape[1])


class SimpleGainSelector(GainSelector):
    """
    Simply choose a single gain channel always.
    """

    channel = traits.Int(default_value=0, help="which gain channel to "
                                               "retain").tag(config=True)

    def select_gains(self, cam_id, multi_gain_waveform):
        return (
            multi_gain_waveform[self.channel],
            (np.ones(multi_gain_waveform.shape[1]) * self.channel).astype(
                np.bool)
        )


class ThresholdGainSelector(GainSelector):
    """
    Select gain channel using fixed-threshold for any sample in the waveform.
    The thresholds are loaded from an `astropy.table.Table` that must contain
    two columns: `cam_id` (the name of the camera) and `gain_threshold_pe`,
    the threshold in photo-electrons per sample at which the switch should
    occur.

    Parameters
    ----------
    threshold_table_name: str
        Name of gain channel table to load
    select_by_sample: bool
        If True, replaces only the waveform samples that are above
        the threshold with low-gain versions, otherwise the full
        low-gain waveform is used.

    Attributes
    ----------
    thresholds: dict
        mapping of cam_id to threshold value
    """

    threshold_table_name = traits.Unicode(
        default_value='gain_channel_thresholds',
        help='Name of gain channel table to load'
    ).tag(config=True)

    select_by_sample = traits.Bool(
        default_value=False,
        help='If True, replaces only the waveform samples that are above '
             'the threshold with low-gain versions, otherwise the full '
             'low-gain waveform is used.'
    ).tag(config=True)

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)

        tab = get_table_dataset(
            self.threshold_table_name,
            role='dl0.tel.svc.gain_thresholds'
        )
        self.thresholds = dict(zip(tab['cam_id'], tab['gain_threshold_pe']))
        self.log.debug("Loaded threshold table: \n %s", tab)

    def __str__(self):
        return f"{self.__class__.__name__}({self.thresholds})"

    def select_gains(self, cam_id, multi_gain_waveform):

        try:
            threshold = self.thresholds[cam_id]
        except KeyError:
            raise KeyError(
                "Camera ID '{}' not found in the gain-threshold "
                "table '{}'".format(cam_id, self.threshold_table_name)
            )

        waveform, gain_mask = pick_gain_channel(
            waveforms=multi_gain_waveform,
            threshold=threshold,
            select_by_sample=self.select_by_sample
        )

        return waveform, gain_mask
