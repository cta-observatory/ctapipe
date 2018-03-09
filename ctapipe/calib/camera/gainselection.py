"""
Algorithms to select correct gain channel
"""
import numpy as np

from ...core import Component, Factory, traits
from ...utils import get_table_dataset

__all__ = ['GainSelectorFactory',
           'ThresholdGainSelector',
           'pick_gain_channel']

def pick_gain_channel(waveforms, threshold, select_by_sample=False):
    """
    the PMTs on some cameras have 2 gain channels. select one
    according to a threshold. ultimately, this will be done IN the
    camera/telescope itself but until then, do it here

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

    global TRUE_FALSE

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
            gain_mask =  (waveforms[0] > threshold).any(axis=1)
            new_waveforms[gain_mask] = waveforms[1][gain_mask]

    elif waveforms.shape[0] == 1:
        new_waveforms = np.squeeze(waveforms)
        gain_mask = np.zeros_like(new_waveforms).astype(bool)
    else:
        raise ValueError("input waveforms has shape %s. not sure what to do "
                         "with that.", waveforms.shape)

    return new_waveforms, gain_mask

class GainSelector(Component):
    """

    """


    def select_gains(self):
        """
        Takes the dl0 channels and
        """
        pass

class NullGainSelector(GainSelector):
    """
    do no gain selection, leaving possibly 2 gain channels at the DL1 level
    """
    def select_gains(self):
        pass

class ThresholdGainSelector(GainSelector):
    """
    Select gain channel using fixed-threshold for any sample in the waveform.
    The thresholds are loaded from an `astropy.table.Table` that must contain
    two columns: `cam_id` (the name of the camera) and `gain_threshold_pe`,
    the threshold in photo-electrons per sample at which the switch should
    occur.
    """

    threshold_table = traits.Unicode(
        'gain_channel_thresholds',
        help='Name of gain channel table to load'
    ).tag(config=True)

    select_partial_waveform = traits.Bool(
        False,
        help = 'If True, replaces only the waveform samples that are above '
               'the threshold with low-gain versions (assuming already PE '
               'calibrated), otherwise the full low-gain waveform is used '
    ).tag(config=True)

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, kwargs=kwargs)
        self._thresholds = get_table_dataset(self.threshold_table ,
                                             role='dl0.tel.svc.gain_thresholds')
        self.log.debug("Loaded threshold table: \n %s", self._thresholds)

    def __str__(self):
        return "{}:\n{}".format(self.__class__.__name__, self._thresholds)

    def select_gains(self):
        pass


class GainSelectorFactory(Factory):
    """
    Factory to obtain a GainSelector
    """
    base = GainSelector
    default = 'ThresholdGainSelector'
    custom_product_help = 'Gain-channel selection scheme to use.'