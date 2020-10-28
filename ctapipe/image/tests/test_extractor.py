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
    FixedWindowSum,
    NeighborPeakWindowSum,
    TwoPassWindowSum,
    FullWaveformSum,
)
from ctapipe.image.toymodel import WaveformModel
from ctapipe.instrument import SubarrayDescription, TelescopeDescription

extractors = non_abstract_children(ImageExtractor)
# FixedWindowSum has no peak finding and need to be set manually
extractors.remove(FixedWindowSum)


@pytest.fixture(scope="module")
def subarray():
    subarray = SubarrayDescription(
        "test array",
        tel_positions={1: np.zeros(3) * u.m, 2: np.zeros(3) * u.m},
        tel_descriptions={
            1: TelescopeDescription.from_name(
                optics_name="SST-ASTRI", camera_name="CHEC"
            ),
            2: TelescopeDescription.from_name(
                optics_name="SST-ASTRI", camera_name="CHEC"
            ),
        },
    )

    # Create reference pulse
    sample_width = 0.5
    reference_pulse_sample_width = sample_width / 10
    reference_pulse_duration = 100
    pulse_sigma = 6
    ref_time = np.arange(0, reference_pulse_duration, reference_pulse_sample_width)
    reference_pulse = norm.pdf(ref_time, reference_pulse_duration / 2, pulse_sigma)

    readout = subarray.tel[1].camera.readout
    readout.reference_pulse_shape = np.array([reference_pulse])
    readout.reference_pulse_sample_width = u.Quantity(
        reference_pulse_sample_width, u.ns
    )
    readout.sampling_rate = u.Quantity(1 / sample_width, u.GHz)
    return subarray


def get_test_toymodel(subarray, minCharge=100, maxCharge=1000):
    telid = list(subarray.tel.keys())[0]
    n_pixels = subarray.tel[telid].camera.geometry.n_pixels
    n_samples = 96
    readout = subarray.tel[telid].camera.readout

    random = np.random.RandomState(1)
    charge = random.uniform(minCharge, maxCharge, n_pixels)
    mid = (n_samples // 2) / readout.sampling_rate.to_value(u.GHz)
    time = random.uniform(mid - 1, mid + 1, n_pixels)

    waveform_model = WaveformModel.from_camera_readout(readout)
    waveform = waveform_model.get_waveform(charge, time, n_samples)

    selected_gain_channel = np.zeros(charge.size, dtype=np.int)

    return waveform, subarray, telid, selected_gain_channel, charge, time


@pytest.fixture(scope="module")
def toymodel(subarray):
    return get_test_toymodel(subarray)


def test_extract_around_peak(toymodel):
    waveforms, _, _, _, _, _ = toymodel
    n_pixels, n_samples = waveforms.shape
    rand = np.random.RandomState(1)
    peak_index = rand.uniform(0, n_samples, n_pixels).astype(np.int)
    charge, peak_time = extract_around_peak(waveforms, peak_index, 7, 3, 1)
    assert (charge >= 0).all()
    assert (peak_time >= 0).all() and (peak_time <= n_samples).all()

    x = np.arange(100)
    y = norm.pdf(x, 41.2, 6)
    charge, peak_time = extract_around_peak(y[np.newaxis, :], 0, x.size, 0, 1)
    assert_allclose(charge[0], 1.0, rtol=1e-3)
    assert_allclose(peak_time[0], 41.2, rtol=1e-3)

    # Test negative amplitude
    y_offset = y - y.max() / 2
    charge, _ = extract_around_peak(y_offset[np.newaxis, :], 0, x.size, 0, 1)
    assert_allclose(charge, y_offset.sum(), rtol=1e-3)
    assert charge.dtype == np.float32


def test_extract_around_peak_charge_expected(toymodel):
    waveforms = np.ones((2048, 96))
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


def test_neighbor_average_waveform(toymodel):
    waveforms, subarray, telid, _, _, _ = toymodel
    neighbors = subarray.tel[telid].camera.geometry.neighbor_matrix_sparse
    average_wf = neighbor_average_waveform(
        waveforms,
        neighbors_indices=neighbors.indices,
        neighbors_indptr=neighbors.indptr,
        lwt=0,
    )

    pixel = 0
    _, nei_pixel = np.where(neighbors[pixel].A)
    expected_average = waveforms[nei_pixel].sum(0) / len(nei_pixel)
    assert_allclose(average_wf[pixel], expected_average, rtol=1e-3)

    lwt = 4
    average_wf = neighbor_average_waveform(
        waveforms,
        neighbors_indices=neighbors.indices,
        neighbors_indptr=neighbors.indptr,
        lwt=lwt,
    )

    pixel = 1
    _, nei_pixel = np.where(neighbors[pixel].A)
    nei_pixel = np.concatenate([nei_pixel, [pixel] * lwt])
    expected_average = waveforms[nei_pixel].sum(0) / len(nei_pixel)
    assert_allclose(average_wf[pixel], expected_average, rtol=1e-3)


def test_extract_peak_time_within_range():
    x = np.arange(100)
    # Generic waveform that goes from positive to negative in window
    # Can cause extreme values with incorrect handling of weighted average
    y = -1.2 * x + 20
    _, peak_time = extract_around_peak(y[np.newaxis, :], 12, 10, 0, 1)
    assert (peak_time >= 0).all() & (peak_time < x.size).all()


def test_baseline_subtractor(toymodel):
    waveforms, _, _, _, _, _ = toymodel
    n_pixels, _ = waveforms.shape
    rand = np.random.RandomState(1)
    offset = np.arange(n_pixels)[:, np.newaxis]
    waveforms = rand.normal(0, 0.1, waveforms.shape) + offset
    assert_allclose(waveforms[3].mean(), 3, rtol=1e-2)
    baseline_subtracted = subtract_baseline(waveforms, 0, 10)
    assert_allclose(baseline_subtracted.mean(), 0, atol=1e-3)


def test_integration_correction(subarray):
    readout = subarray.tel[1].camera.readout
    reference_pulse_shape = readout.reference_pulse_shape
    sample_width_ns = (1 / readout.sampling_rate).to_value(u.ns)
    n_ref_samples = reference_pulse_shape.shape[1]
    sampled = reference_pulse_shape[0].reshape((n_ref_samples // 10, 10)).sum(-1) / 10
    full_integral = np.sum(sampled * sample_width_ns)

    for window_start in range(0, sampled.size):
        for window_end in range(window_start + 1, sampled.size):
            window_width = window_end - window_start
            window_shift = sampled.argmax() - window_start
            correction = integration_correction(
                reference_pulse_shape,
                readout.reference_pulse_sample_width.to_value(u.ns),
                sample_width_ns,
                window_width,
                window_shift,
            )[0]
            window_integral = np.sum(sampled[window_start:window_end] * sample_width_ns)
            if window_integral > 1e-8:  # Avoid floating point resolution limit
                np.testing.assert_allclose(full_integral, window_integral * correction)


def test_integration_correction_outofbounds(subarray):
    readout = subarray.tel[1].camera.readout
    reference_pulse_shape = readout.reference_pulse_shape
    sample_width_ns = (1 / readout.sampling_rate).to_value(u.ns)
    n_ref_samples = reference_pulse_shape.shape[1]
    sampled = reference_pulse_shape[0].reshape((n_ref_samples // 10, 10)).sum(-1) / 10
    full_integral = np.sum(sampled * sample_width_ns)

    for window_start in range(0, sampled.size):
        for window_end in range(sampled.size, sampled.size + 20):
            window_width = window_end - window_start
            window_shift = sampled.argmax() - window_start
            correction = integration_correction(
                reference_pulse_shape,
                readout.reference_pulse_sample_width.to_value(u.ns),
                sample_width_ns,
                window_width,
                window_shift,
            )[0]
            window_integral = np.sum(sampled[window_start:window_end] * sample_width_ns)
            if window_integral > 1e-8:  # Avoid floating point resolution limit
                np.testing.assert_allclose(full_integral, window_integral * correction)


@pytest.mark.parametrize("Extractor", extractors)
def test_extractors(Extractor, toymodel):
    waveforms, subarray, telid, selected_gain_channel, true_charge, true_time = toymodel
    extractor = Extractor(subarray=subarray)
    charge, peak_time = extractor(waveforms, telid, selected_gain_channel)
    assert_allclose(charge, true_charge, rtol=0.1)
    assert_allclose(peak_time, true_time, rtol=0.1)


@pytest.mark.parametrize("Extractor", extractors)
def test_integration_correction_off(Extractor, toymodel):
    # full waveform extractor does not have an integration correction
    if Extractor is FullWaveformSum:
        return

    waveforms, subarray, telid, selected_gain_channel, true_charge, true_time = toymodel
    extractor = Extractor(subarray=subarray, apply_integration_correction=False)
    charge, peak_time = extractor(waveforms, telid, selected_gain_channel)

    # peak time should stay the same
    assert_allclose(peak_time, true_time, rtol=0.1)

    # charge should be too small without correction
    assert np.all(charge <= true_charge)


def test_fixed_window_sum(toymodel):
    waveforms, subarray, telid, selected_gain_channel, true_charge, true_time = toymodel
    extractor = FixedWindowSum(subarray=subarray, peak_index=47)
    charge, peak_time = extractor(waveforms, telid, selected_gain_channel)
    assert_allclose(charge, true_charge, rtol=0.1)
    assert_allclose(peak_time, true_time, rtol=0.1)


def test_neighbor_peak_window_sum_lwt(toymodel):
    waveforms, subarray, telid, selected_gain_channel, true_charge, true_time = toymodel
    extractor = NeighborPeakWindowSum(subarray=subarray, lwt=4)
    assert extractor.lwt.tel[telid] == 4
    charge, peak_time = extractor(waveforms, telid, selected_gain_channel)
    assert_allclose(charge, true_charge, rtol=0.1)
    assert_allclose(peak_time, true_time, rtol=0.1)


def test_two_pass_window_sum(subarray):
    extractor = TwoPassWindowSum(subarray=subarray)
    min_charges = [1, 10, 100]
    max_charges = [10, 100, 1000]
    for minCharge, maxCharge in zip(min_charges, max_charges):
        toymodel = get_test_toymodel(subarray, minCharge, maxCharge)
        (
            waveforms,
            subarray,
            telid,
            selected_gain_channel,
            true_charge,
            true_time,
        ) = toymodel
        charge, pulse_time = extractor(waveforms, telid, selected_gain_channel)
        assert_allclose(charge, true_charge, rtol=0.1)
        assert_allclose(pulse_time, true_time, rtol=0.1)


def test_waveform_extractor_factory(toymodel):
    waveforms, subarray, telid, selected_gain_channel, true_charge, true_time = toymodel
    extractor = ImageExtractor.from_name("LocalPeakWindowSum", subarray=subarray)
    charge, peak_time = extractor(waveforms, telid, selected_gain_channel)
    assert_allclose(charge, true_charge, rtol=0.1)
    assert_allclose(peak_time, true_time, rtol=0.1)


def test_waveform_extractor_factory_args(subarray):
    """
    Config is supposed to be created by a `Tool`
    """
    config = Config({"ImageExtractor": {"window_width": 20, "window_shift": 3}})

    extractor = ImageExtractor.from_name(
        "LocalPeakWindowSum", subarray=subarray, config=config
    )
    assert extractor.window_width.tel[None] == 20
    assert extractor.window_shift.tel[None] == 3

    with pytest.warns(UserWarning):
        ImageExtractor.from_name("FullWaveformSum", config=config, subarray=subarray)


def test_extractor_tel_param(toymodel):
    waveforms, subarray, _, _, _, _ = toymodel
    _, n_samples = waveforms.shape

    config = Config(
        {
            "ImageExtractor": {
                "window_width": [("type", "*", n_samples), ("id", "2", n_samples // 2)],
                "peak_index": 0,
            }
        }
    )

    waveforms, subarray, _, _, _, _ = toymodel
    n_pixels, n_samples = waveforms.shape
    extractor = ImageExtractor.from_name(
        "FixedWindowSum", subarray=subarray, config=config
    )

    assert extractor.peak_index.tel[None] == 0
    assert extractor.peak_index.tel[1] == 0
    assert extractor.peak_index.tel[2] == 0
    assert extractor.window_width.tel[None] == n_samples
    assert extractor.window_width.tel[1] == n_samples
    assert extractor.window_width.tel[2] == n_samples // 2


@pytest.mark.parametrize("Extractor", non_abstract_children(ImageExtractor))
def test_dtype(Extractor, subarray):

    tel_id = 1
    n_pixels = subarray.tel[tel_id].camera.geometry.n_pixels
    selected_gain_channel = np.zeros(n_pixels, dtype=int)

    waveforms = np.ones((n_pixels, 50), dtype="float64")
    extractor = Extractor(subarray=subarray)
    charge, peak_time = extractor(waveforms, tel_id, selected_gain_channel)
    assert charge.dtype == np.float32
    assert peak_time.dtype == np.float32
