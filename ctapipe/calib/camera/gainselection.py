"""
Algorithms to select correct gain channel
"""
import numpy as np

from ...core import Component, Factory, traits
from ...utils import get_table_dataset

__all__ = ['GainSelectorFactory',
           'ThresholdGainSelector',
           'pick_gain_channel']

# use this in the selection of the gain channels
TRUE_FALSE = np.array([[True], [False]])

def pick_gain_channel(waveforms, threshold):
    """
    the PMTs on some cameras have 2 gain channels. select one
    according to a threshold. ultimately, this will be done IN the
    camera/telescope itself but until then, do it here

    Parameters
    ----------
    waveforms
    threshold

    Returns
    -------
    tuple:
        gain-selected intensity, boolean array of which channel was chosen
    """

    global TRUE_FALSE

    if waveforms.shape[0] > 1:
        waveforms = np.squeeze(waveforms)
        pick = (threshold < waveforms).any(axis=0) != TRUE_FALSE
        waveforms = waveforms.T[pick.T]
    else:
        waveforms = np.squeeze(waveforms)
    return waveforms

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

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, kwargs=kwargs)
        self._thresholds = get_table_dataset(self.threshold_table)
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