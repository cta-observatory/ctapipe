import numpy as np
from ctapipe.calib.camera.gainselection import ManualGainSelector, \
    ThresholdGainSelector, GainChannel, GainSelector


class TestGainSelector(GainSelector):
    def select_channel(self, waveforms):
        return GainChannel.HIGH


def test_gain_selector():
    shape = (2, 2048, 128)
    waveforms = np.indices(shape)[1]
    waveforms[1] *= 2

    gain_selector = TestGainSelector()
    waveforms_gs, pixel_channel = gain_selector(waveforms)
    np.testing.assert_equal(waveforms[GainChannel.HIGH], waveforms_gs)
    np.testing.assert_equal(pixel_channel, 0)


def test_pre_selected():
    shape = (2048, 128)
    waveforms = np.zeros(shape)

    gain_selector = TestGainSelector()
    waveforms_gs, pixel_channel = gain_selector(waveforms)
    assert waveforms.shape == waveforms_gs.shape
    assert pixel_channel is None


def test_single_channel():
    shape = (1, 2048, 128)
    waveforms = np.zeros(shape)

    gain_selector = TestGainSelector()
    waveforms_gs, pixel_channel = gain_selector(waveforms)
    assert waveforms_gs.shape == (2048, 128)
    assert (pixel_channel == 0).all()


def test_manual_gain_selector():
    shape = (2, 2048, 128)
    waveforms = np.indices(shape)[1]
    waveforms[1] *= 2

    gs_high = ManualGainSelector(channel="HIGH")
    waveforms_gs, pixel_channel = gs_high(waveforms)
    np.testing.assert_equal(waveforms[GainChannel.HIGH], waveforms_gs)
    np.testing.assert_equal(pixel_channel, 0)

    gs_low = ManualGainSelector(channel="LOW")
    waveforms_gs, pixel_channel = gs_low(waveforms)
    np.testing.assert_equal(waveforms[GainChannel.LOW], waveforms_gs)
    np.testing.assert_equal(pixel_channel, 1)


def test_threshold_gain_selector():
    shape = (2, 2048, 128)
    waveforms = np.zeros(shape)
    waveforms[1] = 1
    waveforms[0, 0] = 100

    gain_selector = ThresholdGainSelector(threshold=50)
    waveforms_gs, pixel_channel = gain_selector(waveforms)
    assert (waveforms_gs[0] == 1).all()
    assert (waveforms_gs[np.arange(1, 2048)] == 0).all()
    assert pixel_channel[0] == 1
    assert (pixel_channel[np.arange(1, 2048)] == 0).all()
