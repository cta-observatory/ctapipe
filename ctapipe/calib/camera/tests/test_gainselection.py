import numpy as np
import pytest

from ctapipe.calib.camera.gainselection import ThresholdGainSelector
from ctapipe.calib.camera.gainselection import SimpleGainSelector
from ctapipe.calib.camera.gainselection import pick_gain_channel


def test_pick_gain_channel():
    threshold = 100
    good_hg_value = 35
    good_lg_value = 50

    dummy_waveforms = np.ones((2, 1000, 30)) * good_lg_value
    dummy_waveforms[1:] = good_hg_value  # high gains

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

    assert gain_mask.shape == (1000,)
    assert new_waveforms.shape == (1000, 30)
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

    assert gain_mask.shape == new_waveforms.shape
    assert new_waveforms.shape == (1000, 30)
    assert (new_waveforms[500:, 13:15] == good_hg_value).all()
    assert (new_waveforms[500:, :13] == good_lg_value).all()
    assert (new_waveforms[500:, 15:] == good_lg_value).all()


def test_pick_gain_channel_bad_input():
    input_waveforms = np.arange(10).reshape(1, 10)
    waveforms, gain_mask = pick_gain_channel(input_waveforms, threshold=4)
    assert gain_mask is not None
    assert (waveforms == input_waveforms).all()


def test_threshold_gain_selector():
    selector = ThresholdGainSelector()
    print(selector)

    assert 'NectarCam' in selector.thresholds

    threshold = selector.thresholds['NectarCam']
    good_hg_value = 35
    good_lg_value = 50
    dummy_waveforms = np.ones((2, 1000, 30)) * good_lg_value
    dummy_waveforms[1:] = good_hg_value  #
    dummy_waveforms[0, 500:, 13:15] = threshold + 10

    new_waveforms, gain_mask = selector.select_gains("NectarCam",
                                                     dummy_waveforms)
    assert gain_mask.shape == (1000,)
    assert new_waveforms.shape == (1000, 30)
    assert (new_waveforms[500:] == good_hg_value).all()
    assert (new_waveforms[:500] == good_lg_value).all()

    selector.select_by_sample = True

    new_waveforms, gain_mask = selector.select_gains("NectarCam",
                                                     dummy_waveforms)

    assert new_waveforms.shape == (1000, 30)
    assert (new_waveforms[500:, 13:15] == good_hg_value).all()
    assert (new_waveforms[500:, :13] == good_lg_value).all()
    assert (new_waveforms[500:, 15:] == good_lg_value).all()
    assert gain_mask.shape == new_waveforms.shape

    # test some failures:
    # Camera that doesn't have a threshold:
    with pytest.raises(KeyError):
        selector.select_gains("NonExistantCamera", dummy_waveforms)

    # 3-gain channel input:
    with pytest.raises(ValueError):
        selector.select_gains("NectarCam", np.ones((3, 1000, 30)))

    # 1-gain channel input:
    wf0 = np.ones((1, 1000, 1))
    wf1, gm = selector.select_gains("ASTRICam", wf0)
    assert wf1.shape == (1000,)
    assert gm.shape == (1000,)


def test_simple_gain_selector():
    gs = SimpleGainSelector()

    for chan in [0, 1]:
        gs.channel = chan

        waveforms_2g = np.random.normal(size=(2, 1000, 30))
        waveforms_1g, mask = gs.select_gains("NectarCam", waveforms_2g)

        assert waveforms_1g.shape == (1000, 30)
        assert (waveforms_1g == waveforms_2g[chan]).all()
        assert mask.shape == (1000,)
