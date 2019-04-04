import pytest
import numpy as np
from scipy.stats import norm
from numpy.testing import assert_allclose
from ctapipe.instrument import CameraGeometry
from ctapipe.image.extractor import (
    extract_charge_from_peakpos_array,
    neighbor_average_waveform,
    extract_pulse_time_weighted_average,
    subtract_baseline,
    ImageExtractor,
    FullWaveformSum,
    FixedWindowSum,
    GlobalPeakWindowSum,
    LocalPeakWindowSum,
    NeighborPeakWindowSum,
    BaselineSubtractedNeighborPeakWindowSum,
)


@pytest.fixture(scope='module')
def camera_waveforms():
    camera = CameraGeometry.from_name("CHEC")

    n_pixels = camera.n_pixels
    n_samples = 96
    mid = n_samples // 2
    pulse_sigma = 6
    r_hi = np.random.RandomState(1)
    r_lo = np.random.RandomState(2)

    x = np.arange(n_samples)

    # Randomize times
    t_pulse_hi = r_hi.uniform(mid - 10, mid + 10, n_pixels)[:, np.newaxis]
    t_pulse_lo = r_lo.uniform(mid + 10, mid + 20, n_pixels)[:, np.newaxis]

    # Create pulses
    y_hi = norm.pdf(x, t_pulse_hi, pulse_sigma)
    y_lo = norm.pdf(x, t_pulse_lo, pulse_sigma)

    # Randomize amplitudes
    y_hi *= r_hi.uniform(100, 1000, n_pixels)[:, np.newaxis]
    y_lo *= r_lo.uniform(100, 1000, n_pixels)[:, np.newaxis]

    y = np.stack([y_hi, y_lo])

    return y, camera


def test_extract_charge_from_peakpos_array(camera_waveforms):
    waveforms, _ = camera_waveforms
    _, n_pixels, n_samples = waveforms.shape
    rand = np.random.RandomState(1)
    peakpos = rand.uniform(0, n_samples, (2, n_pixels)).astype(np.int)
    charge = extract_charge_from_peakpos_array(waveforms, peakpos, 7, 3)

    assert_allclose(charge[0][0], 146.022991, rtol=1e-3)
    assert_allclose(charge[1][0], 22.393974, rtol=1e-3)


def test_neighbor_average_waveform(camera_waveforms):
    waveforms, camera = camera_waveforms
    nei = camera.neighbor_matrix_where
    average_wf = neighbor_average_waveform(waveforms, nei, 0)

    assert_allclose(average_wf[0, 0, 48], 28.690154, rtol=1e-3)
    assert_allclose(average_wf[1, 0, 48], 2.221035, rtol=1e-3)

    average_wf = neighbor_average_waveform(waveforms, nei, 4)

    assert_allclose(average_wf[0, 0, 48], 98.565743, rtol=1e-3)
    assert_allclose(average_wf[1, 0, 48], 9.578896, rtol=1e-3)


def test_extract_pulse_time_weighted_average(camera_waveforms):
    waveforms, _ = camera_waveforms
    pulse_time = extract_pulse_time_weighted_average(waveforms)

    assert_allclose(pulse_time[0][0], 46.34044, rtol=1e-3)
    assert_allclose(pulse_time[1][0], 62.359948, rtol=1e-3)


def test_baseline_subtractor(camera_waveforms):
    waveforms, _ = camera_waveforms
    n_chan, n_pixels, n_samples = waveforms.shape
    rand = np.random.RandomState(1)
    offset = np.arange(n_pixels)[np.newaxis, :, np.newaxis]
    waveforms = rand.normal(0, 0.1, waveforms.shape) + offset
    assert_allclose(waveforms[0, 3].mean(), 3, rtol=1e-2)
    baseline_subtracted = subtract_baseline(waveforms, 0, 10)
    assert_allclose(baseline_subtracted.mean(), 0, atol=1e-3)


def test_full_waveform_sum(camera_waveforms):
    waveforms, _ = camera_waveforms
    extractor = FullWaveformSum()
    charge, pulse_time = extractor(waveforms)

    assert_allclose(charge[0][0], 545.945, rtol=1e-3)
    assert_allclose(charge[1][0], 970.025, rtol=1e-3)
    assert_allclose(pulse_time[0][0], 46.34044, rtol=1e-3)
    assert_allclose(pulse_time[1][0], 62.359948, rtol=1e-3)


def test_fixed_window_sum(camera_waveforms):
    waveforms, _ = camera_waveforms
    extractor = FixedWindowSum(window_start=45)
    charge, pulse_time = extractor(waveforms)

    assert_allclose(charge[0][0], 232.559, rtol=1e-3)
    assert_allclose(charge[1][0], 32.539, rtol=1e-3)
    assert_allclose(pulse_time[0][0], 46.34044, rtol=1e-3)
    assert_allclose(pulse_time[1][0], 62.359948, rtol=1e-3)


def test_global_peak_window_sum(camera_waveforms):
    waveforms, _ = camera_waveforms
    extractor = GlobalPeakWindowSum()
    charge, pulse_time = extractor(waveforms)

    assert_allclose(charge[0][0], 232.559, rtol=1e-3)
    assert_allclose(charge[1][0], 425.406, rtol=1e-3)
    assert_allclose(pulse_time[0][0], 46.34044, rtol=1e-3)
    assert_allclose(pulse_time[1][0], 62.359948, rtol=1e-3)


def test_local_peak_window_sum(camera_waveforms):
    waveforms, _ = camera_waveforms
    extractor = LocalPeakWindowSum()
    charge, pulse_time = extractor(waveforms)

    assert_allclose(charge[0][0], 240.3, rtol=1e-3)
    assert_allclose(charge[1][0], 427.158, rtol=1e-3)
    assert_allclose(pulse_time[0][0], 46.34044, rtol=1e-3)
    assert_allclose(pulse_time[1][0], 62.359948, rtol=1e-3)


def test_neighbor_peak_window_sum(camera_waveforms):
    waveforms, camera = camera_waveforms
    nei = camera.neighbor_matrix_where
    extractor = NeighborPeakWindowSum()
    extractor.neighbors = nei
    charge, pulse_time = extractor(waveforms)

    assert_allclose(charge[0][0], 94.671, rtol=1e-3)
    assert_allclose(charge[1][0], 426.887, rtol=1e-3)
    assert_allclose(pulse_time[0][0], 46.34044, rtol=1e-3)
    assert_allclose(pulse_time[1][0], 62.359948, rtol=1e-3)

    extractor.lwt = 4
    charge, pulse_time = extractor(waveforms)

    assert_allclose(charge[0][0], 220.418657, rtol=1e-3)
    assert_allclose(charge[1][0], 426.887, rtol=1e-3)
    assert_allclose(pulse_time[0][0], 46.34044, rtol=1e-3)
    assert_allclose(pulse_time[1][0], 62.359948, rtol=1e-3)


def test_baseline_subtracted_neighbor_peak_window_sum(camera_waveforms):
    waveforms, camera = camera_waveforms
    nei = camera.neighbor_matrix_where
    extractor = BaselineSubtractedNeighborPeakWindowSum()
    extractor.neighbors = nei
    charge, pulse_time = extractor(waveforms)

    assert_allclose(charge[0][0], 94.671, rtol=1e-3)
    assert_allclose(charge[1][0], 426.887, rtol=1e-3)
    assert_allclose(pulse_time[0][0], 46.34044, rtol=1e-3)
    assert_allclose(pulse_time[1][0], 62.359948, rtol=1e-3)


def test_waveform_extractor_factory(camera_waveforms):
    waveforms, _ = camera_waveforms
    extractor = ImageExtractor.from_name('LocalPeakWindowSum')
    extractor(waveforms)


def test_waveform_extractor_factory_args():
    """
    Config is supposed to be created by a `Tool`
    """
    from traitlets.config.loader import Config
    config = Config(
        {
            'ImageExtractor': {
                'window_width': 20,
                'window_shift': 3,
            }
        }
    )

    extractor = ImageExtractor.from_name(
        'LocalPeakWindowSum',
        config=config,
    )
    assert extractor.window_width == 20
    assert extractor.window_shift == 3

    with pytest.warns(UserWarning):
        ImageExtractor.from_name(
            'FullWaveformSum',
            config=config,
        )
