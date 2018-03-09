import numpy as np

from ctapipe.calib.camera.gainselection import GainSelectorFactory
from ctapipe.calib.camera.gainselection import ThresholdGainSelector
from ctapipe.calib.camera.gainselection import pick_gain_channel



def test_pick_gain_channel():

    threshold = 100
    good_hg_value = 35

    dummy_waveforms = np.ones((2,1000,30)) * 50
    dummy_waveforms[1:] = good_hg_value # high gains

    # set pixels  above 500's sample 13 to a high value (to trigger switch)
    dummy_waveforms[0, 500:, 13] = threshold + 10

    # at the end, the final waveforms should be of shape (1000,30) and
    # pixels from 500 and above should have the low-gain value of good_hg_value

    new_waveforms = pick_gain_channel(waveforms=dummy_waveforms,
                                      threshold=threshold)

    assert new_waveforms.shape == (1000,30)
    assert all(new_waveforms[500:] == good_hg_value)