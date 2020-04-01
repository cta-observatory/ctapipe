import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from scipy.stats import norm
from traitlets.config.loader import Config

from ctapipe.core import non_abstract_children
from ctapipe.image.extractor import (
    extract_around_peak,
    neighbor_average_waveform,
    subtract_baseline,
    integration_correction,
    ImageExtractor,
    FullWaveformSum,
    FixedWindowSum,
    GlobalPeakWindowSum,
    LocalPeakWindowSum,
    NeighborPeakWindowSum,
    BaselineSubtractedNeighborPeakWindowSum,
    TwoPassWindowSum,
)
from ctapipe.instrument import SubarrayDescription, TelescopeDescription

extractors = non_abstract_children(ImageExtractor)
# FixedWindowSum has no peak finding and need to be set manually
extractors.remove(FixedWindowSum)


@pytest.fixture(scope="module")
def camera_waveforms():
    telid = 1
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
        },
    )

    n_pixels = subarray.tel[1].camera.geometry.n_pixels
    n_samples = 96
    mid = n_samples // 2
    pulse_sigma = 6
    random = np.random.RandomState(1)

    x = np.arange(n_samples)

    # Randomize times
    t_pulse = random.uniform(mid - 1, mid + 1, n_pixels)[:, np.newaxis]

    # Create pulses
    y = norm.pdf(x, t_pulse, pulse_sigma)

    # Create reference pulse
    x_ref = np.arange(n_samples * 2)
    reference_pulse = norm.pdf(x_ref, n_samples, pulse_sigma * 2)
    readout = subarray.tel[telid].camera.readout
    readout.reference_pulse_shape = np.array([reference_pulse])
    readout.reference_pulse_sample_width = u.Quantity(0.5, u.ns)

    # Randomize amplitudes
    charge = random.uniform(100, 1000, n_pixels)
    y *= charge[:, np.newaxis]

    selected_gain_channel = np.zeros(n_pixels, dtype=np.int)

    return y, subarray, telid, selected_gain_channel, charge


@pytest.fixture("module")
def reference_pulse():
    reference_pulse_step = 0.09
    n_reference_pulse_samples = 1280
    reference_pulse_shape = np.array(
        [
            norm.pdf(np.arange(n_reference_pulse_samples), 600, 100) * 1.7,
            norm.pdf(np.arange(n_reference_pulse_samples), 700, 100) * 1.7,
        ]
    )
    return reference_pulse_shape, reference_pulse_step


@pytest.fixture("module")
def sampled_reference_pulse(reference_pulse):
    reference_pulse_shape, reference_pulse_step = reference_pulse
    n_channels, n_reference_pulse_samples = reference_pulse_shape.shape
    pulse_max_sample = n_reference_pulse_samples * reference_pulse_step
    sample_width_ns = 2
    pulse_shape_x = np.arange(0, pulse_max_sample, reference_pulse_step)
    sampled_edges = np.arange(0, pulse_max_sample, sample_width_ns)
    sampled_pulse = np.array(
        [
            np.histogram(
                pulse_shape_x,
                sampled_edges,
                weights=reference_pulse_shape[ichan],
                density=True,
            )[0]
            for ichan in range(n_channels)
        ]
    )
    return sampled_pulse, sample_width_ns


def test_extract_around_peak(camera_waveforms):
    waveforms, subarray, telid, selected_gain_channel, true_charge = camera_waveforms
    n_pixels, n_samples = waveforms.shape
    rand = np.random.RandomState(1)
    peak_index = rand.uniform(0, n_samples, n_pixels).astype(np.int)
    charge, pulse_time = extract_around_peak(waveforms, peak_index, 7, 3, 1)
    assert_allclose(charge[0], 112.184183, rtol=1e-3)
    assert_allclose(pulse_time[0], 40.789745, rtol=1e-3)

    x = np.arange(100)
    y = norm.pdf(x, 41.2, 6)
    charge, pulse_time = extract_around_peak(y[np.newaxis, :], 0, x.size, 0, 1)
    assert_allclose(charge[0], 1.0, rtol=1e-3)
    assert_allclose(pulse_time[0], 41.2, rtol=1e-3)

    # Test negative amplitude
    y_offset = y - y.max() / 2
    charge, pulse_time = extract_around_peak(y_offset[np.newaxis, :], 0, x.size, 0, 1)
    assert_allclose(charge, y_offset.sum(), rtol=1e-3)


def test_extract_around_peak_charge_expected(camera_waveforms):
    waveforms, subarray, telid, selected_gain_channel, true_charge = camera_waveforms
    waveforms = np.ones(waveforms.shape)
    n_samples = waveforms.shape[-1]
    sampling_rate_ghz = 1

    peak_index = 0
    width = 10
    shift = 0
    charge, _ = extract_around_peak(
        waveforms, peak_index, width, shift, sampling_rate_ghz
    )
    assert_equal(charge, 10)

    peak_index = 0
    width = 10
    shift = 10
    charge, _ = extract_around_peak(
        waveforms, peak_index, width, shift, sampling_rate_ghz
    )
    assert_equal(charge, 0)

    peak_index = 0
    width = 20
    shift = 10
    charge, _ = extract_around_peak(
        waveforms, peak_index, width, shift, sampling_rate_ghz
    )
    assert_equal(charge, 10)

    peak_index = n_samples
    width = 10
    shift = 0
    charge, _ = extract_around_peak(
        waveforms, peak_index, width, shift, sampling_rate_ghz
    )
    assert_equal(charge, 0)

    peak_index = n_samples
    width = 20
    shift = 10
    charge, _ = extract_around_peak(
        waveforms, peak_index, width, shift, sampling_rate_ghz
    )
    assert_equal(charge, 10)

    peak_index = 0
    width = n_samples * 3
    shift = n_samples
    charge, _ = extract_around_peak(
        waveforms, peak_index, width, shift, sampling_rate_ghz
    )
    assert_equal(charge, n_samples)

    # Test sampling rate
    peak_index = n_samples
    width = 20
    shift = 10
    charge, _ = extract_around_peak(
        waveforms, peak_index, width, shift, sampling_rate_ghz * 2
    )
    assert_equal(charge, 5)


def test_neighbor_average_waveform(camera_waveforms):
    waveforms, subarray, telid, selected_gain_channel, true_charge = camera_waveforms
    nei = subarray.tel[1].camera.geometry.neighbor_matrix_where
    average_wf = neighbor_average_waveform(waveforms, nei, 0)
    assert_allclose(average_wf[0, 48], 51.089826, rtol=1e-3)

    average_wf = neighbor_average_waveform(waveforms, nei, 4)
    assert_allclose(average_wf[0, 48], 123.662305, rtol=1e-3)


def test_extract_pulse_time_within_range():
    x = np.arange(100)
    # Generic waveform that goes from positive to negative in window
    # Can cause extreme values with incorrect handling of weighted average
    y = -1.2 * x + 20
    _, pulse_time = extract_around_peak(y[np.newaxis, :], 12, 10, 0, 1)
    assert (pulse_time >= 0).all() & (pulse_time < x.size).all()


def test_baseline_subtractor(camera_waveforms):
    waveforms, _, _, _, _ = camera_waveforms
    n_pixels, _ = waveforms.shape
    rand = np.random.RandomState(1)
    offset = np.arange(n_pixels)[:, np.newaxis]
    waveforms = rand.normal(0, 0.1, waveforms.shape) + offset
    assert_allclose(waveforms[3].mean(), 3, rtol=1e-2)
    baseline_subtracted = subtract_baseline(waveforms, 0, 10)
    assert_allclose(baseline_subtracted.mean(), 0, atol=1e-3)


def test_integration_correction(reference_pulse, sampled_reference_pulse):
    reference_pulse_shape, reference_pulse_step = reference_pulse
    sampled_pulse, sample_width_ns = sampled_reference_pulse
    sampled_pulse_fc = sampled_pulse[0]  # Test first channel
    full_integral = np.sum(sampled_pulse[0] * sample_width_ns)

    for window_start in range(0, sampled_pulse_fc.size):
        for window_end in range(window_start + 1, sampled_pulse_fc.size):
            window_width = window_end - window_start
            window_shift = sampled_pulse_fc.argmax() - window_start
            correction = integration_correction(
                reference_pulse_shape,
                reference_pulse_step,
                sample_width_ns,
                window_width,
                window_shift,
            )[0]
            window_integral = np.sum(
                sampled_pulse_fc[window_start:window_end] * sample_width_ns
            )
            np.testing.assert_allclose(full_integral, window_integral * correction)


def test_integration_correction_outofbounds(reference_pulse, sampled_reference_pulse):
    reference_pulse_shape, reference_pulse_step = reference_pulse
    sampled_pulse, sample_width_ns = sampled_reference_pulse
    sampled_pulse_fc = sampled_pulse[0]  # Test first channel
    full_integral = np.sum(sampled_pulse[0] * sample_width_ns)

    for window_start in range(0, sampled_pulse_fc.size):
        for window_end in range(sampled_pulse_fc.size, sampled_pulse_fc.size + 20):
            window_width = window_end - window_start
            window_shift = sampled_pulse_fc.argmax() - window_start
            correction = integration_correction(
                reference_pulse_shape,
                reference_pulse_step,
                sample_width_ns,
                window_width,
                window_shift,
            )[0]
            window_integral = np.sum(
                sampled_pulse_fc[window_start:window_end] * sample_width_ns
            )
            np.testing.assert_allclose(full_integral, window_integral * correction)


@pytest.mark.parametrize("Extractor", extractors)
def test_extractors(Extractor, camera_waveforms):
    waveforms, subarray, telid, selected_gain_channel, true_charge = camera_waveforms
    extractor = Extractor(subarray=subarray)
    charge, pulse_time = extractor(waveforms, telid, selected_gain_channel)
    assert_allclose(charge, true_charge, rtol=0.1)
    assert_allclose(pulse_time, waveforms.shape[1] // 2, rtol=0.1)


def test_fixed_window_sum(camera_waveforms):
    waveforms, subarray, telid, selected_gain_channel, true_charge = camera_waveforms
    extractor = FixedWindowSum(subarray=subarray, window_start=48)
    charge, pulse_time = extractor(waveforms, telid, selected_gain_channel)
    assert_allclose(charge, true_charge, rtol=0.1)
    assert_allclose(pulse_time, waveforms.shape[1] // 2, rtol=0.1)


def test_neighbor_peak_window_sum_lwt(camera_waveforms):
    waveforms, subarray, telid, selected_gain_channel, true_charge = camera_waveforms
    extractor = NeighborPeakWindowSum(subarray=subarray, lwt=4)
    assert extractor.lwt.tel[telid] == 4
    charge, pulse_time = extractor(waveforms, telid, selected_gain_channel)
    assert_allclose(charge, true_charge, rtol=0.1)
    assert_allclose(pulse_time, waveforms.shape[1] // 2, rtol=0.1)


def test_two_pass_window_sum(camera_waveforms):
    waveforms, subarray = camera_waveforms
    extractor = TwoPassWindowSum(subarray=subarray)
    charge, pulse_time = extractor(waveforms, telid=1)
    assert_allclose(charge[0], 176.307343, rtol=1e-3)
    assert_allclose(pulse_time[0], 46.018546, rtol=1e-3)


def test_waveform_extractor_factory(camera_waveforms):
    waveforms, subarray, telid, selected_gain_channel, true_charge = camera_waveforms
    extractor = ImageExtractor.from_name("LocalPeakWindowSum", subarray=subarray)
    charge, pulse_time = extractor(waveforms, telid, selected_gain_channel)
    assert_allclose(charge, true_charge, rtol=0.1)
    assert_allclose(pulse_time, waveforms.shape[1] // 2, rtol=0.1)


def test_waveform_extractor_factory_args(camera_waveforms):
    """
    Config is supposed to be created by a `Tool`
    """
    _, subarray, _, _, _ = camera_waveforms
    config = Config({"ImageExtractor": {"window_width": 20, "window_shift": 3}})

    extractor = ImageExtractor.from_name(
        "LocalPeakWindowSum", subarray=subarray, config=config
    )
    assert extractor.window_width.tel[None] == 20
    assert extractor.window_shift.tel[None] == 3

    with pytest.warns(UserWarning):
        ImageExtractor.from_name("FullWaveformSum", config=config, subarray=subarray)


def test_extractor_tel_param(camera_waveforms):
    waveforms, subarray, _, _, _ = camera_waveforms
    _, n_samples = waveforms.shape

    config = Config(
        {
            "ImageExtractor": {
                "window_width": [("type", "*", n_samples), ("id", "2", n_samples // 2)],
                "window_start": 0,
            }
        }
    )

    waveforms, subarray, _, _, _ = camera_waveforms
    n_pixels, n_samples = waveforms.shape
    extractor = ImageExtractor.from_name(
        "FixedWindowSum", subarray=subarray, config=config
    )

    assert extractor.window_start.tel[None] == 0
    assert extractor.window_start.tel[1] == 0
    assert extractor.window_start.tel[2] == 0
    assert extractor.window_width.tel[None] == n_samples
    assert extractor.window_width.tel[1] == n_samples
    assert extractor.window_width.tel[2] == n_samples // 2
