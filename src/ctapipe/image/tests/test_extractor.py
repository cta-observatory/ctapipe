from copy import deepcopy

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from scipy.signal import filtfilt
from scipy.stats import norm
from traitlets.config.loader import Config
from traitlets.traitlets import TraitError

from ctapipe.core import non_abstract_children
from ctapipe.image.cleaning import dilate
from ctapipe.image.extractor import (
    FixedWindowSum,
    FlashCamExtractor,
    FullWaveformSum,
    ImageExtractor,
    NeighborPeakWindowSum,
    SlidingWindowMaxSum,
    TwoPassWindowSum,
    VarianceExtractor,
    __filtfilt_fast,
    adaptive_centroid,
    deconvolve,
    extract_around_peak,
    extract_sliding_window,
    integration_correction,
    neighbor_average_maximum,
    subtract_baseline,
)
from ctapipe.image.toymodel import SkewedGaussian, WaveformModel, obtain_time_image
from ctapipe.instrument import SubarrayDescription
from ctapipe.io import EventSource

extractors = non_abstract_children(ImageExtractor)
# FixedWindowSum has no peak finding and need to be set manually
extractors.remove(FixedWindowSum)
extractors.remove(FlashCamExtractor)

camera_toymodels = ["toymodel", "toymodel_sst", "toymodel_lst", "toymodel_mst_fc"]


@pytest.fixture(scope="module")
def subarray(prod5_sst, reference_location):
    subarray = SubarrayDescription(
        "test array",
        tel_positions={1: np.zeros(3) * u.m, 2: np.zeros(3) * u.m},
        tel_descriptions={
            1: deepcopy(prod5_sst),
            2: deepcopy(prod5_sst),
        },
        reference_location=reference_location,
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
    readout.n_samples = 96
    return subarray


@pytest.fixture(scope="module")
def subarray_2_sst(prod5_sst, reference_location):
    subarray = SubarrayDescription(
        "test array",
        tel_positions={1: np.zeros(3) * u.m, 2: np.zeros(3) * u.m},
        tel_descriptions={
            1: prod5_sst,
            2: prod5_sst,
        },
        reference_location=reference_location,
    )
    return subarray


@pytest.fixture(scope="module")
def subarray_1_lst(prod3_lst, reference_location):
    subarray = SubarrayDescription(
        "One LST",
        tel_positions={1: np.zeros(3) * u.m},
        tel_descriptions={1: prod3_lst},
        reference_location=reference_location,
    )
    return subarray


@pytest.fixture(scope="module")
def subarray_1_mst_fc(prod5_mst_flashcam, reference_location):
    subarray = SubarrayDescription(
        "One MST with FlashCam",
        tel_positions={1: np.zeros(3) * u.m},
        tel_descriptions={1: prod5_mst_flashcam},
        reference_location=reference_location,
    )
    return subarray


def get_test_toymodel(subarray, minCharge=100, maxCharge=1000):
    tel_id = list(subarray.tel.keys())[0]
    n_pixels = subarray.tel[tel_id].camera.geometry.n_pixels
    readout = subarray.tel[tel_id].camera.readout
    n_samples = readout.n_samples

    random = np.random.RandomState(1)
    charge = random.uniform(minCharge, maxCharge, n_pixels)
    mid = (n_samples // 2) / readout.sampling_rate.to_value(u.GHz)
    time = random.uniform(mid - 1, mid + 1, n_pixels)

    waveform_model = WaveformModel.from_camera_readout(readout)
    waveform = waveform_model.get_waveform(charge, time, n_samples)

    selected_gain_channel = np.zeros(charge.size, dtype=np.int64)
    if waveform.shape[-3] != 1:
        selected_gain_channel = None

    return waveform, subarray, tel_id, selected_gain_channel, charge, time


@pytest.fixture(scope="module")
def toymodel(subarray):
    return get_test_toymodel(subarray)


@pytest.fixture(scope="module")
def toymodel_sst(subarray_2_sst):
    return get_test_toymodel(subarray_2_sst)


@pytest.fixture(scope="module")
def toymodel_lst(subarray_1_lst):
    return get_test_toymodel(subarray_1_lst)


@pytest.fixture(scope="module")
def toymodel_mst_fc(subarray_1_mst_fc):
    return get_test_toymodel(subarray_1_mst_fc)


def get_test_toymodel_gradient(subarray, minCharge=100, maxCharge=1000):
    tel_id = list(subarray.tel.keys())[0]
    n_pixels = subarray.tel[tel_id].camera.geometry.n_pixels
    geometry = subarray.tel[tel_id].camera.geometry
    n_samples = 96
    readout = subarray.tel[tel_id].camera.readout

    random = np.random.RandomState(1)
    charge = random.uniform(minCharge, maxCharge, n_pixels)
    tmid = (n_samples // 2) / readout.sampling_rate.to_value(u.GHz)
    tmax = (n_samples - 1) / readout.sampling_rate.to_value(u.GHz)
    time = random.uniform(0, tmax, n_pixels)
    pix_id = 1
    mask = geometry.pix_id == pix_id

    dilated_mask = mask.copy()
    for _ in range(4):
        dilated_mask = dilate(geometry, dilated_mask)

    x_d = subarray.tel[tel_id].camera.geometry.pix_x.value
    min_xd = np.min(x_d[dilated_mask])
    diff_xd = x_d[dilated_mask] - min_xd
    slope = 15  # ns
    intercept = 0  # ns
    time[dilated_mask] = tmid + (slope * diff_xd + intercept)
    waveform_model = WaveformModel.from_camera_readout(readout)
    waveform = waveform_model.get_waveform(charge, time, n_samples)

    selected_gain_channel = np.zeros(charge.size, dtype=np.int64)

    return waveform, subarray, tel_id, selected_gain_channel, charge, time, dilated_mask


@pytest.fixture(scope="module")
def toymodel_mst_fc_time(subarray_1_mst_fc: object) -> object:
    return get_test_toymodel_gradient(subarray_1_mst_fc)


def test_extract_around_peak(toymodel):
    waveforms, _, _, _, _, _ = toymodel
    _, n_pixels, n_samples = waveforms.shape
    rand = np.random.RandomState(1)
    peak_index = rand.uniform(0, n_samples, n_pixels).astype(np.int64)
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


def test_extract_sliding_window(toymodel):
    waveforms, _, _, _, _, _ = toymodel
    n_samples = waveforms.shape[-1]
    charge, peak_time = extract_sliding_window(waveforms, 7, 1)
    assert (charge >= 0).all()
    assert (peak_time >= 0).all() and (peak_time <= n_samples).all()

    x = np.arange(100)
    y = norm.pdf(x, 41.2, 6)
    charge, peak_time = extract_sliding_window(y[np.newaxis, :], x.size, 1)
    assert_allclose(charge[0], 1.0, rtol=1e-3)
    assert_allclose(peak_time[0], 41.2, rtol=1e-3)

    # Test negative amplitude
    y_offset = y - y.max() / 2
    charge, _ = extract_sliding_window(y_offset[np.newaxis, :], x.size, 1)
    assert_allclose(charge, y_offset.sum(), rtol=1e-3)
    assert charge.dtype == np.float32


def test_extract_around_peak_charge_expected():
    waveforms = np.ones((1, 2048, 96))
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


def test_variance_extractor(toymodel):
    _, subarray, _, _, _, _ = toymodel
    # make dummy data with known variance
    rng = np.random.default_rng(0)
    var_data = rng.normal(2.0, 5.0, size=(2, 1855, 5000))
    extractor = ImageExtractor.from_name("VarianceExtractor", subarray=subarray)

    variance = extractor(var_data, 0, None, None).image
    np.testing.assert_allclose(variance, np.var(var_data, axis=2), rtol=1e-3)


@pytest.mark.parametrize("toymodels", camera_toymodels)
def test_neighbor_average_peakpos(toymodels, request):
    waveforms, subarray, tel_id, _, _, _ = request.getfixturevalue(toymodels)
    n_channels, n_pixels, _ = waveforms.shape
    neighbors = subarray.tel[tel_id].camera.geometry.neighbor_matrix_sparse
    broken_pixels = np.zeros((n_channels, n_pixels), dtype=bool)
    peak_pos = neighbor_average_maximum(
        waveforms,
        neighbors_indices=neighbors.indices,
        neighbors_indptr=neighbors.indptr,
        local_weight=0,
        broken_pixels=broken_pixels,
    )

    pixel = 0
    _, nei_pixel = np.nonzero(neighbors[pixel].toarray())
    expected_average = waveforms[:, nei_pixel].sum(1) / len(nei_pixel)
    expected_peak_pos = np.argmax(expected_average, axis=-1)
    for ichannel in range(waveforms.shape[-3]):
        assert peak_pos[ichannel][pixel] == expected_peak_pos[ichannel]

    local_weight = 4
    peak_pos = neighbor_average_maximum(
        waveforms,
        neighbors_indices=neighbors.indices,
        neighbors_indptr=neighbors.indptr,
        local_weight=local_weight,
        broken_pixels=broken_pixels,
    )

    pixel = 1
    _, nei_pixel = np.nonzero(neighbors[pixel].toarray())
    nei_pixel = np.concatenate([nei_pixel, [pixel] * local_weight])
    expected_average = waveforms[:, nei_pixel].sum(1) / len(nei_pixel)
    expected_peak_pos = np.argmax(expected_average, axis=-1)
    for ichannel in range(waveforms.shape[-3]):
        assert peak_pos[ichannel][pixel] == expected_peak_pos[ichannel]


def test_extract_peak_time_within_range():
    x = np.arange(100)
    # Generic waveform that goes from positive to negative in window
    # Can cause extreme values with incorrect handling of weighted average
    y = -1.2 * x + 20
    _, peak_time = extract_around_peak(y[np.newaxis, :], 12, 10, 0, 1)
    assert (peak_time >= 0).all() & (peak_time < x.size).all()


def test_baseline_subtractor(toymodel):
    waveforms, _, _, _, _, _ = toymodel
    n_pixels = waveforms.shape[-2]
    rand = np.random.RandomState(1)
    offset = np.arange(n_pixels)[:, np.newaxis]
    waveforms = rand.normal(0, 0.1, waveforms.shape) + offset
    assert_allclose(waveforms[0][3].mean(), 3, rtol=1e-2)
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


@pytest.mark.parametrize("toymodels", camera_toymodels)
@pytest.mark.parametrize("Extractor", extractors)
def test_extractors(Extractor, toymodels, request):
    (
        waveforms,
        subarray,
        tel_id,
        selected_gain_channel,
        true_charge,
        true_time,
    ) = request.getfixturevalue(toymodels)
    extractor = Extractor(subarray=subarray)
    n_channels, n_pixels, _ = waveforms.shape
    broken_pixels = np.zeros((n_channels, n_pixels), dtype=bool)

    if Extractor is TwoPassWindowSum and waveforms.shape[-3] != 1:
        with pytest.raises(AttributeError):
            extractor(waveforms, tel_id, selected_gain_channel, broken_pixels)
        return

    if Extractor is VarianceExtractor:
        return

    dl1 = extractor(waveforms, tel_id, selected_gain_channel, broken_pixels)
    assert dl1.is_valid
    if dl1.image.ndim == 1:
        assert_allclose(dl1.image, true_charge, rtol=0.2)
        assert_allclose(dl1.peak_time, true_time, rtol=0.2)
    else:
        for ichannel in range(dl1.image.shape[-2]):
            assert_allclose(dl1.image[ichannel], true_charge, rtol=0.2)
            assert_allclose(dl1.peak_time[ichannel], true_time, rtol=0.2)


@pytest.mark.parametrize("toymodels", camera_toymodels)
@pytest.mark.parametrize("Extractor", extractors)
def test_integration_correction_off(Extractor, toymodels, request):
    # full waveform extractor does not have an integration correction
    if Extractor in (FullWaveformSum, VarianceExtractor):
        return

    (
        waveforms,
        subarray,
        tel_id,
        selected_gain_channel,
        true_charge,
        true_time,
    ) = request.getfixturevalue(toymodels)
    extractor = Extractor(subarray=subarray, apply_integration_correction=False)
    n_channels, n_pixels, _ = waveforms.shape
    broken_pixels = np.zeros((n_channels, n_pixels), dtype=bool)

    if Extractor is TwoPassWindowSum and waveforms.shape[-3] != 1:
        with pytest.raises(AttributeError):
            extractor(waveforms, tel_id, selected_gain_channel, broken_pixels)
        return

    dl1 = extractor(waveforms, tel_id, selected_gain_channel, broken_pixels)
    assert dl1.is_valid

    # peak time should stay the same
    # charge should be too small without correction for the used reference pulse
    # shapes (not in general).
    if dl1.image.ndim == 1:
        assert np.all(dl1.image <= true_charge)
        assert_allclose(dl1.peak_time, true_time, rtol=0.1)
    else:
        for ichannel in range(dl1.image.shape[-2]):
            assert np.all(dl1.image[ichannel] <= true_charge)
            assert_allclose(dl1.peak_time[ichannel], true_time, rtol=0.1)


def test_fixed_window_sum(toymodel):
    (
        waveforms,
        subarray,
        tel_id,
        selected_gain_channel,
        true_charge,
        true_time,
    ) = toymodel

    extractor = FixedWindowSum(subarray=subarray, peak_index=47)
    n_channels, n_pixels, _ = waveforms.shape
    broken_pixels = np.zeros((n_channels, n_pixels), dtype=bool)
    dl1 = extractor(waveforms, tel_id, selected_gain_channel, broken_pixels)
    assert dl1.is_valid
    assert_allclose(dl1.image, true_charge, rtol=0.1)
    assert_allclose(dl1.peak_time, true_time, rtol=0.1)

    waveforms = np.append(waveforms, waveforms, axis=-3)
    dl1 = extractor(waveforms, tel_id, None, broken_pixels)
    for ichannel in range(dl1.image.shape[-2]):
        assert_allclose(dl1.image[ichannel], true_charge, rtol=0.1)
        assert_allclose(dl1.peak_time[ichannel], true_time, rtol=0.1)


@pytest.mark.parametrize("toymodels", camera_toymodels)
def test_sliding_window_max_sum(toymodels, request):
    (
        waveforms,
        subarray,
        tel_id,
        selected_gain_channel,
        true_charge,
        true_time,
    ) = request.getfixturevalue(toymodels)
    extractor = SlidingWindowMaxSum(subarray=subarray)
    n_channels, n_pixels, _ = waveforms.shape
    broken_pixels = np.zeros((n_channels, n_pixels), dtype=bool)
    dl1 = extractor(waveforms, tel_id, selected_gain_channel, broken_pixels)
    assert dl1.is_valid
    if dl1.image.ndim == 1:
        assert_allclose(dl1.image, true_charge, rtol=0.1)
        assert_allclose(dl1.peak_time, true_time, rtol=0.1)
    else:
        for ichannel in range(dl1.image.shape[-2]):
            assert_allclose(dl1.image[ichannel], true_charge, rtol=0.1)
            assert_allclose(dl1.peak_time[ichannel], true_time, rtol=0.1)


def test_neighbor_peak_window_sum_local_weight(toymodel):
    (
        waveforms,
        subarray,
        tel_id,
        selected_gain_channel,
        true_charge,
        true_time,
    ) = toymodel
    extractor = NeighborPeakWindowSum(subarray=subarray, local_weight=4)
    assert extractor.local_weight.tel[tel_id] == 4
    n_channels, n_pixels, _ = waveforms.shape
    broken_pixels = np.zeros((n_channels, n_pixels), dtype=bool)
    dl1 = extractor(waveforms, tel_id, selected_gain_channel, broken_pixels)
    assert_allclose(dl1.image, true_charge, rtol=0.1)
    assert_allclose(dl1.peak_time, true_time, rtol=0.1)
    assert dl1.is_valid


def test_Two_pass_window_sum_no_noise(subarray_1_lst):
    rng = np.random.default_rng(0)

    subarray = subarray_1_lst

    camera = subarray.tel[1].camera
    geometry = camera.geometry
    readout = camera.readout
    sampling_rate = readout.sampling_rate.to_value("GHz")
    n_samples = 30  # LSTCam & NectarCam specific
    max_time_readout = (n_samples / sampling_rate) * u.ns

    # True image settings
    x = 0.0 * u.m
    y = 0.0 * u.m
    length = 0.2 * u.m
    width = 0.05 * u.m
    psi = 45.0 * u.deg
    skewness = 0.0
    # build the true time evolution in a way that
    # the whole image is about the readout window
    time_gradient = u.Quantity(max_time_readout.value / length.value, u.ns / u.m)
    time_intercept = u.Quantity(max_time_readout.value / 2, u.ns)
    intensity = 600
    nsb_level_pe = 0

    # create the image
    m = SkewedGaussian(x, y, length, width, psi, skewness)
    true_charge, true_signal, true_noise = m.generate_image(
        geometry, intensity=intensity, nsb_level_pe=nsb_level_pe, rng=rng
    )
    signal_pixels = true_signal > 2
    # create a pulse-times image without noise
    # we can make new functions later
    time_noise = rng.uniform(0, 0, geometry.n_pixels)
    time_signal = obtain_time_image(
        geometry.pix_x, geometry.pix_y, x, y, psi, time_gradient, time_intercept
    )

    true_charge[(time_signal < 0) | (time_signal > (n_samples / sampling_rate))] = 0

    true_time = np.average(
        np.column_stack([time_noise, time_signal]),
        weights=np.column_stack([true_noise, true_signal]) + 1,
        axis=1,
    )

    # Define the model for the waveforms to fill with the information from
    # the simulated image
    waveform_model = WaveformModel.from_camera_readout(readout, gain_channel="HIGH")
    waveforms = waveform_model.get_waveform(true_charge, true_time, n_samples)
    selected_gain_channel = np.zeros(true_charge.size, dtype=np.int64)

    # Define the extractor
    extractor = TwoPassWindowSum(subarray=subarray)

    # Select the signal pixels for which the integration window is well inside
    # the readout window (in this case we should require more accuracy)
    true_peaks = np.rint(true_time * sampling_rate).astype(np.int64)

    # integration of 5 samples centered on peak + 1 sample of error
    # to not be really on the edge
    min_good_sample = 2 + 1
    max_good_sample = n_samples - 1 - min_good_sample
    integration_window_inside = (true_peaks >= min_good_sample) & (
        true_peaks < max_good_sample
    )

    # Test only the 1st pass
    extractor.disable_second_pass = True
    n_channels, n_pixels, _ = waveforms.shape
    broken_pixels = np.zeros((n_channels, n_pixels), dtype=bool)
    dl1_pass1 = extractor(waveforms, 1, selected_gain_channel, broken_pixels)
    assert_allclose(
        dl1_pass1.image[signal_pixels & integration_window_inside],
        true_charge[signal_pixels & integration_window_inside],
        rtol=0.15,
    )
    assert_allclose(
        dl1_pass1.peak_time[signal_pixels & integration_window_inside],
        true_time[signal_pixels & integration_window_inside],
        rtol=0.15,
    )

    # Test also the 2nd pass
    extractor.disable_second_pass = False
    dl1_pass2 = extractor(waveforms, 1, selected_gain_channel, broken_pixels)

    # Check that we have gained signal charge by using the 2nd pass
    # This also checks that the test image has triggered the 2nd pass
    # (i.e. it is not so bad to have < 3 pixels in the preliminary cleaned image)
    reco_charge1 = np.sum(dl1_pass1.image[signal_pixels & integration_window_inside])
    reco_charge2 = np.sum(dl1_pass2.image[signal_pixels & integration_window_inside])
    # since there is no noise in this test, 1st pass will find the peak and 2nd
    # can at most do the same
    assert (reco_charge2 / reco_charge1) < 1
    assert dl1_pass1.is_valid

    # Test only signal pixels for which it is expected to find most of the
    # charge well inside the readout window
    assert_allclose(
        dl1_pass2.image[signal_pixels & integration_window_inside],
        true_charge[signal_pixels & integration_window_inside],
        rtol=0.3,
        atol=2.0,
    )
    assert_allclose(
        dl1_pass2.peak_time[signal_pixels & integration_window_inside],
        true_time[signal_pixels & integration_window_inside],
        rtol=0.3,
        atol=2.0,
    )


def test_waveform_extractor_factory(toymodel):
    (
        waveforms,
        subarray,
        tel_id,
        selected_gain_channel,
        true_charge,
        true_time,
    ) = toymodel
    extractor = ImageExtractor.from_name("LocalPeakWindowSum", subarray=subarray)

    n_channels, n_pixels, _ = waveforms.shape
    broken_pixels = np.zeros((n_channels, n_pixels), dtype=bool)
    dl1 = extractor(waveforms, tel_id, selected_gain_channel, broken_pixels)
    assert_allclose(dl1.image, true_charge, rtol=0.1)
    assert_allclose(dl1.peak_time, true_time, rtol=0.1)


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

    # this basically tests that traitlets do not accept unknown traits,
    # which is tested for all traitlets in the core tests already
    with pytest.raises(TraitError):
        ImageExtractor.from_name("FullWaveformSum", config=config, subarray=subarray)


def test_extractor_tel_param(toymodel):
    waveforms, subarray, _, _, _, _ = toymodel
    n_samples = waveforms.shape[-1]

    config = Config(
        {
            "ImageExtractor": {
                "window_width": [("type", "*", n_samples), ("id", "2", n_samples // 2)],
                "peak_index": 0,
            }
        }
    )

    waveforms, subarray, _, _, _, _ = toymodel
    _, _, n_samples = waveforms.shape
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

    waveforms = np.ones((1, n_pixels, 50), dtype="float64")
    extractor = Extractor(subarray=subarray)
    n_channels, n_pixels, _ = waveforms.shape
    broken_pixels = np.zeros((n_channels, n_pixels), dtype=bool)
    dl1 = extractor(waveforms, tel_id, selected_gain_channel, broken_pixels)

    if Extractor is not VarianceExtractor:
        assert dl1.peak_time.dtype == np.float32

    assert dl1.image.dtype == np.float32


def test_global_peak_window_sum_with_pixel_fraction(subarray):
    from ctapipe.image.extractor import GlobalPeakWindowSum

    tel_id = 1
    camera = subarray.tel[tel_id].camera
    sample_rate = camera.readout.sampling_rate.to_value(u.ns**-1)
    n_pixels = camera.geometry.n_pixels
    selected_gain_channel = np.zeros(n_pixels, dtype=np.uint8)

    bright_pixels = np.zeros(n_pixels, dtype=bool)
    bright_pixels[np.random.choice(n_pixels, size=int(0.1 * n_pixels))] = True

    # signal in dim pixels is in slice 10, signal in bright pixels is in slice 30
    waveforms = np.zeros((1, n_pixels, 50), dtype="float64")
    waveforms[:, ~bright_pixels, 9] = 3
    waveforms[:, ~bright_pixels, 10] = 5
    waveforms[:, ~bright_pixels, 11] = 2
    waveforms[:, bright_pixels, 29] = 5
    waveforms[:, bright_pixels, 30] = 10
    waveforms[:, bright_pixels, 31] = 3

    extractor = GlobalPeakWindowSum(
        subarray=subarray,
        window_width=8,
        window_shift=4,
        pixel_fraction=0.05,
        apply_integration_correction=False,
    )

    n_channels, n_pixels, _ = waveforms.shape
    broken_pixels = np.zeros((n_channels, n_pixels), dtype=bool)
    dl1 = extractor(waveforms, tel_id, selected_gain_channel, broken_pixels)

    assert np.allclose(dl1.image[bright_pixels], 18)
    assert np.allclose(dl1.image[~bright_pixels], 0)

    expected = np.average([29, 30, 31], weights=[5, 10, 3])
    assert np.allclose(dl1.peak_time[bright_pixels], expected / sample_rate)


def test_adaptive_centroid(toymodel_mst_fc):
    waveforms, subarray, tel_id, _, _, _ = toymodel_mst_fc

    neighbors = subarray.tel[tel_id].camera.geometry.neighbor_matrix_sparse
    n_channels, n_pixels, _ = waveforms.shape
    broken_pixels = np.zeros((n_channels, n_pixels), dtype=bool)

    trig_time = np.argmax(waveforms, axis=-1)
    peak_time = adaptive_centroid(
        waveforms,
        trig_time,
        1,
    )

    assert (peak_time == trig_time).all()

    waveforms = waveforms[np.min(waveforms, axis=-1) > 0.0].reshape((0, 0, 0))
    peak_pos = neighbor_average_maximum(
        waveforms,
        neighbors_indices=neighbors.indices,
        neighbors_indptr=neighbors.indptr,
        local_weight=0,
        broken_pixels=broken_pixels,
    )

    peak_time = adaptive_centroid(
        waveforms,
        peak_pos,
        0.0,
    )
    assert (peak_pos == peak_time).all()


def test_deconvolve(toymodel_mst_fc):
    waveforms, _, _, _, _, _ = toymodel_mst_fc

    deconvolved_waveforms_0 = deconvolve(waveforms, 0, 0, 0.0)

    assert (deconvolved_waveforms_0[..., 1:] == waveforms[..., 1:]).all()

    deconvolved_waveforms_1 = deconvolve(waveforms, 0, 0, 1.0)

    assert (deconvolved_waveforms_1[..., 1:] == np.diff(waveforms, axis=-1)).all()


def test_upsampling(toymodel_mst_fc):
    waveforms, _, _, _, _, _ = toymodel_mst_fc
    upsampling_even = 4
    upsampling_odd = 3
    filt_even = np.ones(upsampling_even)
    filt_weighted_even = filt_even / upsampling_even
    signal_even = np.repeat(waveforms, upsampling_even, axis=-1)
    up_waveforms_even = __filtfilt_fast(signal_even, filt_weighted_even)

    np.testing.assert_allclose(
        up_waveforms_even,
        filtfilt(
            np.ones(upsampling_even),
            upsampling_even,
            np.repeat(waveforms, upsampling_even, axis=-1),
        ),
        rtol=1e-4,
        atol=1e-4,
    )

    filt_odd = np.ones(upsampling_odd)
    filt_weighted_odd = filt_odd / upsampling_odd
    signal_odd = np.repeat(waveforms, upsampling_odd, axis=-1)
    up_waveforms_odd = __filtfilt_fast(signal_odd, filt_weighted_odd)

    np.testing.assert_allclose(
        up_waveforms_odd,
        filtfilt(
            np.ones(upsampling_odd),
            upsampling_odd,
            np.repeat(waveforms, upsampling_odd, axis=-1),
        ),
        rtol=1e-4,
        atol=1e-4,
    )


def test_FC_time(toymodel_mst_fc_time):
    # Test time on toy model with time gradient (other toy not sensitive to timing bugs!!)
    (
        waveforms,
        subarray,
        tel_id,
        selected_gain_channel,
        _,
        true_time,
        mask,
    ) = toymodel_mst_fc_time

    n_channels, n_pixels, _ = waveforms.shape
    broken_pixels = np.zeros((n_channels, n_pixels), dtype=bool)

    extractor = FlashCamExtractor(subarray=subarray, leading_edge_timing=True)
    dl1 = extractor(waveforms, tel_id, selected_gain_channel, broken_pixels)
    assert_allclose(dl1.peak_time[mask], true_time[mask], rtol=0.1)
    assert dl1.is_valid

    extractor = FlashCamExtractor(
        subarray=subarray, upsampling=1, leading_edge_timing=True
    )
    dl1 = extractor(waveforms, tel_id, selected_gain_channel, broken_pixels)
    assert_allclose(dl1.peak_time[mask], true_time[mask], rtol=0.1)
    assert dl1.is_valid

    extractor = FlashCamExtractor(
        subarray=subarray, upsampling=1, leading_edge_timing=False
    )
    dl1 = extractor(waveforms, tel_id, selected_gain_channel, broken_pixels)
    assert_allclose(dl1.peak_time[mask], true_time[mask], rtol=0.1)
    assert dl1.is_valid

    extractor = FlashCamExtractor(subarray=subarray, leading_edge_timing=False)
    dl1 = extractor(waveforms, tel_id, selected_gain_channel, broken_pixels)
    assert_allclose(dl1.peak_time[mask], true_time[mask], rtol=0.1)
    assert dl1.is_valid


def test_flashcam_extractor(toymodel_mst_fc, prod5_gamma_simtel_path):
    # Test charge on standard toy model
    (
        waveforms,
        subarray,
        tel_id,
        selected_gain_channel,
        true_charge,
        _,
    ) = toymodel_mst_fc
    extractor = FlashCamExtractor(subarray=subarray, leading_edge_timing=True)
    n_channels, n_pixels, _ = waveforms.shape
    broken_pixels = np.zeros((n_channels, n_pixels), dtype=bool)
    dl1 = extractor(waveforms, tel_id, selected_gain_channel, broken_pixels)
    assert_allclose(dl1.image, true_charge, rtol=0.1)
    assert dl1.is_valid

    # Test on prod5 simulations
    with EventSource(prod5_gamma_simtel_path) as source:
        subarray = source.subarray
        extractor = FlashCamExtractor(subarray)

        def is_flashcam(tel_id):
            return subarray.tel[tel_id].camera.name == "FlashCam"

        for event in source:
            for tel_id in filter(is_flashcam, event.trigger.tels_with_trigger):
                true_charge = event.simulation.tel[tel_id].true_image

                waveforms = event.r1.tel[tel_id].waveform
                selected_gain_channel = np.zeros(waveforms.shape[-2], dtype=np.int64)
                broken_pixels = event.mon.tel[
                    tel_id
                ].pixel_status.hardware_failing_pixels

                dl1 = extractor(waveforms, tel_id, selected_gain_channel, broken_pixels)
                assert dl1.is_valid

                bright_pixels = (
                    (true_charge > 30) & (true_charge < 3000) & (~broken_pixels[0])
                )
                assert_allclose(
                    dl1.image[bright_pixels], true_charge[bright_pixels], rtol=0.35
                )
