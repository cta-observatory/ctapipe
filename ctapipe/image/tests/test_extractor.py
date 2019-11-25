import pytest
import numpy as np
from scipy.stats import norm
from numpy.testing import assert_allclose, assert_equal
import astropy.units as u
from ctapipe.instrument import SubarrayDescription, TelescopeDescription
from ctapipe.image.extractor import (
    extract_around_peak,
    neighbor_average_waveform,
    subtract_baseline,
    ImageExtractor,
    FullWaveformSum,
    FixedWindowSum,
    GlobalPeakWindowSum,
    LocalPeakWindowSum,
    NeighborPeakWindowSum,
    BaselineSubtractedNeighborPeakWindowSum,
)
from traitlets.config.loader import Config


@pytest.fixture(scope='module')
def camera_waveforms():
    subarray = SubarrayDescription(
        "test array",
        tel_positions={1: np.zeros(3) * u.m, 2: np.ones(3) * u.m},
        tel_descriptions={
            1: TelescopeDescription.from_name(
                optics_name="SST-ASTRI", camera_name="CHEC"
            ),
            2: TelescopeDescription.from_name(
                optics_name="SST-ASTRI", camera_name="CHEC"
            ),
        }
    )

    n_pixels = subarray.tel[1].camera.n_pixels
    n_samples = 96
    mid = n_samples // 2
    pulse_sigma = 6
    random = np.random.RandomState(1)

    x = np.arange(n_samples)

    # Randomize times
    t_pulse = random.uniform(mid - 10, mid + 10, n_pixels)[:, np.newaxis]

    # Create pulses
    y = norm.pdf(x, t_pulse, pulse_sigma)

    # Randomize amplitudes
    y *= random.uniform(100, 1000, n_pixels)[:, np.newaxis]

    return y, subarray


def test_extract_around_peak(camera_waveforms):
    waveforms, _ = camera_waveforms
    n_pixels, n_samples = waveforms.shape
    rand = np.random.RandomState(1)
    peak_index = rand.uniform(0, n_samples, n_pixels).astype(np.int)
    charge, pulse_time = extract_around_peak(waveforms, peak_index, 7, 3)
    assert_allclose(charge[0], 146.022991, rtol=1e-3)
    assert_allclose(pulse_time[0], 40.659884, rtol=1e-3)

    x = np.arange(100)
    y = norm.pdf(x, 41.2, 6)
    charge, pulse_time = extract_around_peak(y[np.newaxis, :], 0, x.size, 0)
    assert_allclose(charge[0], 1.0, rtol=1e-3)
    assert_allclose(pulse_time[0], 41.2, rtol=1e-3)

    # Test negative amplitude
    y_offset = y - y.max() / 2
    charge, pulse_time = extract_around_peak(y_offset[np.newaxis, :], 0, x.size, 0)
    assert_allclose(charge, y_offset.sum(), rtol=1e-3)


def test_extract_around_peak_charge_expected(camera_waveforms):
    waveforms, _ = camera_waveforms
    waveforms = np.ones(waveforms.shape)
    n_samples = waveforms.shape[-1]

    peak_index = 0
    width = 10
    shift = 0
    charge, _ = extract_around_peak(waveforms, peak_index, width, shift)
    assert_equal(charge, 10)

    peak_index = 0
    width = 10
    shift = 10
    charge, _ = extract_around_peak(waveforms, peak_index, width, shift)
    assert_equal(charge, 0)

    peak_index = 0
    width = 20
    shift = 10
    charge, _ = extract_around_peak(waveforms, peak_index, width, shift)
    assert_equal(charge, 10)

    peak_index = n_samples
    width = 10
    shift = 0
    charge, _ = extract_around_peak(waveforms, peak_index, width, shift)
    assert_equal(charge, 0)

    peak_index = n_samples
    width = 20
    shift = 10
    charge, _ = extract_around_peak(waveforms, peak_index, width, shift)
    assert_equal(charge, 10)

    peak_index = 0
    width = n_samples*3
    shift = n_samples
    charge, _ = extract_around_peak(waveforms, peak_index, width, shift)
    assert_equal(charge, n_samples)


def test_neighbor_average_waveform(camera_waveforms):
    waveforms, subarray = camera_waveforms
    nei = subarray.tel[1].camera.neighbor_matrix_where
    average_wf = neighbor_average_waveform(waveforms, nei, 0)
    assert_allclose(average_wf[0, 48], 28.690154, rtol=1e-3)

    average_wf = neighbor_average_waveform(waveforms, nei, 4)
    assert_allclose(average_wf[0, 48], 98.565743, rtol=1e-3)


def test_extract_pulse_time_within_range():
    x = np.arange(100)
    # Generic waveform that goes from positive to negative in window
    # Can cause extreme values with incorrect handling of weighted average
    y = -1.2 * x + 20
    _, pulse_time = extract_around_peak(
        y[np.newaxis, :], 12, 10, 0
    )
    assert (pulse_time >= 0).all() & (pulse_time < x.size).all()


def test_baseline_subtractor(camera_waveforms):
    waveforms, _ = camera_waveforms
    n_pixels, _ = waveforms.shape
    rand = np.random.RandomState(1)
    offset = np.arange(n_pixels)[:, np.newaxis]
    waveforms = rand.normal(0, 0.1, waveforms.shape) + offset
    assert_allclose(waveforms[3].mean(), 3, rtol=1e-2)
    baseline_subtracted = subtract_baseline(waveforms, 0, 10)
    assert_allclose(baseline_subtracted.mean(), 0, atol=1e-3)


def test_full_waveform_sum(camera_waveforms):
    waveforms, _ = camera_waveforms
    extractor = FullWaveformSum()
    charge, pulse_time = extractor(waveforms)
    assert_allclose(charge[0], 545.945, rtol=1e-3)
    assert_allclose(pulse_time[0], 46.34044, rtol=1e-3)


def test_fixed_window_sum(camera_waveforms):
    waveforms, _ = camera_waveforms
    extractor = FixedWindowSum(window_start=45)
    charge, pulse_time = extractor(waveforms)
    assert_allclose(charge[0], 232.559, rtol=1e-3)
    assert_allclose(pulse_time[0], 47.823488, rtol=1e-3)


def test_global_peak_window_sum(camera_waveforms):
    waveforms, _ = camera_waveforms
    extractor = GlobalPeakWindowSum()
    charge, pulse_time = extractor(waveforms)
    assert_allclose(charge[0], 232.559, rtol=1e-3)
    assert_allclose(pulse_time[0], 47.823488, rtol=1e-3)


def test_local_peak_window_sum(camera_waveforms):
    waveforms, _ = camera_waveforms
    extractor = LocalPeakWindowSum()
    charge, pulse_time = extractor(waveforms)
    assert_allclose(charge[0], 240.3, rtol=1e-3)
    assert_allclose(pulse_time[0], 46.036266, rtol=1e-3)


def test_neighbor_peak_window_sum(camera_waveforms):
    waveforms, subarray = camera_waveforms
    extractor = NeighborPeakWindowSum(subarray=subarray)
    charge, pulse_time = extractor(waveforms, telid=1)
    assert_allclose(charge[0], 94.671, rtol=1e-3)
    assert_allclose(pulse_time[0], 54.116092, rtol=1e-3)

    extractor.lwt = 4
    charge, pulse_time = extractor(waveforms, telid=1)
    assert_allclose(charge[0], 220.418657, rtol=1e-3)
    assert_allclose(pulse_time[0], 48.717848, rtol=1e-3)


def test_baseline_subtracted_neighbor_peak_window_sum(camera_waveforms):
    waveforms, subarray = camera_waveforms
    extractor = BaselineSubtractedNeighborPeakWindowSum(subarray=subarray)
    charge, pulse_time = extractor(waveforms, telid=1)
    assert_allclose(charge[0], 94.671, rtol=1e-3)
    assert_allclose(pulse_time[0], 54.116092, rtol=1e-3)


def test_waveform_extractor_factory(camera_waveforms):
    waveforms, _ = camera_waveforms
    extractor = ImageExtractor.from_name('LocalPeakWindowSum')
    extractor(waveforms)


def test_waveform_extractor_factory_args():
    """
    Config is supposed to be created by a `Tool`
    """
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
    assert extractor.window_width[None] == 20
    assert extractor.window_shift[None] == 3

    with pytest.warns(UserWarning):
        ImageExtractor.from_name(
            'FullWaveformSum',
            config=config,
        )


def test_extractor_tel_param(camera_waveforms):
    waveforms, subarray = camera_waveforms
    _, n_samples = waveforms.shape

    config = Config({
        'ImageExtractor': {
            'window_width': [("type", "*", n_samples), ("id", "2", n_samples//2)],
            'window_start': 0,
        }
    })

    waveforms, subarray = camera_waveforms
    n_pixels, n_samples = waveforms.shape
    extractor = ImageExtractor.from_name("FixedWindowSum", config=config)

    with pytest.raises(KeyError):
        assert extractor.window_width[1] == n_samples

    with pytest.raises(KeyError):
        assert extractor.window_width[2] == n_samples // 2

    assert extractor.window_start[None] == 0
    assert extractor.window_width[None] == n_samples

    extractor = ImageExtractor.from_name(
        "FixedWindowSum", config=config, subarray=subarray
    )

    assert extractor.window_start[1] == 0
    assert extractor.window_start[2] == 0
    assert extractor.window_width[None] == n_samples
    assert extractor.window_width[1] == n_samples
    assert extractor.window_width[2] == n_samples // 2
