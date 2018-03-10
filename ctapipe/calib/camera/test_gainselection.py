import numpy as np
import pytest

from ctapipe.calib.camera.gainselection import GainSelectorFactory
from ctapipe.calib.camera.gainselection import ThresholdGainSelector
from ctapipe.calib.camera.gainselection import pick_gain_channel



def test_pick_gain_channel():

    threshold = 100
    good_hg_value = 35
    good_lg_value = 50

    dummy_waveforms = np.ones((2,1000,30)) * good_lg_value
    dummy_waveforms[1:] = good_hg_value # high gains

    # set pixels  above 500's sample 13 to a high value (to trigger switch)
    dummy_waveforms[0, 500:, 13:15] = threshold + 10

    # First test the default mode of replacing the full waveform:
    # at the end, the final waveforms should be of shape (1000,30) and
    # pixels from 500 and above should have the low-gain value of good_hg_value

    new_waveforms, gain_mask = pick_gain_channel(
        waveforms=dummy_waveforms,
        threshold=threshold,
        select_by_sample=False
    )

    assert new_waveforms.shape == (1000,30)
    assert (new_waveforms[500:] == good_hg_value).all()
    assert (new_waveforms[:500] == good_lg_value).all()

    # Next test the optional mode of replacing only the samples above threshold:
    # at the end, the final waveforms should be of shape (1000,30) and
    # pixels from 500 and above and with sample number 15:13 should have the
    # low-gain value of good_hg_value:

    new_waveforms, gain_mask = pick_gain_channel(
        waveforms=dummy_waveforms,
        threshold=threshold,
        select_by_sample=True
    )

    assert new_waveforms.shape == (1000,30)
    assert (new_waveforms[500:, 13:15] == good_hg_value).all()
    assert (new_waveforms[500:, :13] == good_lg_value).all()
    assert (new_waveforms[500:, 15:] == good_lg_value).all()

def test_pick_gain_channel_bad_input():

    input_waveforms = np.arange(10).reshape(1,10)
    waveforms, gain_mask = pick_gain_channel(input_waveforms, threshold=4)
    assert (waveforms == input_waveforms).all()


@pytest.fixture()
def gain_selector_instance():
    yield ThresholdGainSelector()


def test_gain_selector(gain_selector_instance):
    print(gain_selector_instance)

    assert 'NectarCam' in gain_selector_instance.thresholds


