"""
Tests for CameraCalibrator and related functions
"""
import numpy as np
import pytest
from scipy.stats import norm
from traitlets.config.configurable import Config

from ctapipe.calib.camera.calibrator import (
    CameraCalibrator,
    integration_correction,
)
from ctapipe.image.extractor import LocalPeakWindowSum, FullWaveformSum
from ctapipe.instrument import CameraGeometry
from ctapipe.io.containers import DataContainer


@pytest.fixture(scope="function")
def subarray(example_event):
    return example_event.inst.subarray


@pytest.fixture('module')
def reference_pulse():
    reference_pulse_step = 0.09
    n_reference_pulse_samples = 1280
    reference_pulse_shape = np.array([
        norm.pdf(np.arange(n_reference_pulse_samples), 600, 100) * 1.7,
        norm.pdf(np.arange(n_reference_pulse_samples), 700, 100) * 1.7,
    ])
    return reference_pulse_shape, reference_pulse_step


@pytest.fixture('module')
def sampled_reference_pulse(reference_pulse):
    reference_pulse_shape, reference_pulse_step = reference_pulse
    n_channels, n_reference_pulse_samples = reference_pulse_shape.shape
    pulse_max_sample = n_reference_pulse_samples * reference_pulse_step
    sample_width = 2
    pulse_shape_x = np.arange(0, pulse_max_sample, reference_pulse_step)
    sampled_edges = np.arange(0, pulse_max_sample, sample_width)
    sampled_pulse = np.array([np.histogram(
        pulse_shape_x, sampled_edges, weights=reference_pulse_shape[ichan], density=True
    )[0] for ichan in range(n_channels)])
    return sampled_pulse, sample_width


def test_camera_calibrator(example_event, subarray):
    telid = list(example_event.r0.tel)[0]
    calibrator = CameraCalibrator(subarray=subarray)
    calibrator(example_event)
    image = example_event.dl1.tel[telid].image
    pulse_time = example_event.dl1.tel[telid].pulse_time
    assert image is not None
    assert pulse_time is not None
    assert image.shape == (1764,)
    assert pulse_time.shape == (1764,)


def test_manual_extractor(subarray):
    calibrator = CameraCalibrator(
        subarray=subarray,
        image_extractor=LocalPeakWindowSum(subarray=subarray)
    )
    assert isinstance(calibrator.image_extractor, LocalPeakWindowSum)


def test_config(subarray):
    window_shift = 3
    window_width = 9
    config = Config(
        {
            "LocalPeakWindowSum": {
                "window_shift": window_shift,
                "window_width": window_width,
            }
        }
    )
    calibrator = CameraCalibrator(
        subarray=subarray,
        image_extractor=LocalPeakWindowSum(subarray=subarray, config=config),
        config=config
    )
    assert calibrator.image_extractor.window_shift.tel[None] == window_shift
    assert calibrator.image_extractor.window_width.tel[None] == window_width


def test_integration_correction(reference_pulse, sampled_reference_pulse):
    reference_pulse_shape, reference_pulse_step = reference_pulse
    sampled_pulse, sample_width = sampled_reference_pulse
    sampled_pulse_fc = sampled_pulse[0]  # Test first channel
    full_integral = np.sum(sampled_pulse[0] * sample_width)

    for window_start in range(0, sampled_pulse_fc.size):
        for window_end in range(window_start+1, sampled_pulse_fc.size):
            window_width = window_end - window_start
            window_shift = sampled_pulse_fc.argmax() - window_start
            correction = integration_correction(
                reference_pulse_shape,
                reference_pulse_step, sample_width,
                window_width, window_shift
            )[0]
            window_integral = np.sum(
                sampled_pulse_fc[window_start:window_end] * sample_width
            )
            np.testing.assert_allclose(full_integral, window_integral * correction)


def test_integration_correction_outofbounds(reference_pulse, sampled_reference_pulse):
    reference_pulse_shape, reference_pulse_step = reference_pulse
    sampled_pulse, sample_width = sampled_reference_pulse
    sampled_pulse_fc = sampled_pulse[0]  # Test first channel
    full_integral = np.sum(sampled_pulse[0] * sample_width)

    for window_start in range(0, sampled_pulse_fc.size):
        for window_end in range(sampled_pulse_fc.size, sampled_pulse_fc.size+20):
            window_width = window_end - window_start
            window_shift = sampled_pulse_fc.argmax() - window_start
            correction = integration_correction(
                reference_pulse_shape,
                reference_pulse_step, sample_width,
                window_width, window_shift
            )[0]
            window_integral = np.sum(
                sampled_pulse_fc[window_start:window_end] * sample_width
            )
            np.testing.assert_allclose(full_integral, window_integral * correction)


def test_integration_correction_no_ref_pulse(example_event, subarray):
    telid = list(example_event.r0.tel)[0]
    delattr(example_event, "mc")
    calibrator = CameraCalibrator(subarray=subarray)
    calibrator._calibrate_dl0(example_event, telid)
    correction = calibrator._get_correction(example_event, telid)
    assert (correction == 1).all()


def test_check_r1_empty(example_event, subarray):
    calibrator = CameraCalibrator(subarray=subarray)
    telid = list(example_event.r0.tel)[0]
    waveform = example_event.r1.tel[telid].waveform.copy()
    with pytest.warns(UserWarning):
        example_event.r1.tel[telid].waveform = None
        calibrator._calibrate_dl0(example_event, telid)
        assert example_event.dl0.tel[telid].waveform is None

    assert calibrator._check_r1_empty(None) is True
    assert calibrator._check_r1_empty(waveform) is False

    calibrator = CameraCalibrator(
        subarray=subarray,
        image_extractor=FullWaveformSum(subarray=subarray)
    )
    event = DataContainer()
    event.dl0.tel[telid].waveform = np.full((2048, 128), 2)
    with pytest.warns(UserWarning):
        calibrator(event)
    assert (event.dl0.tel[telid].waveform == 2).all()
    assert (event.dl1.tel[telid].image == 2 * 128).all()


def test_check_dl0_empty(example_event):
    calibrator = CameraCalibrator(subarray=subarray)
    telid = list(example_event.r0.tel)[0]
    calibrator._calibrate_dl0(example_event, telid)
    waveform = example_event.dl0.tel[telid].waveform.copy()
    with pytest.warns(UserWarning):
        example_event.dl0.tel[telid].waveform = None
        calibrator._calibrate_dl1(example_event, telid)
        assert example_event.dl1.tel[telid].image is None

    assert calibrator._check_dl0_empty(None) is True
    assert calibrator._check_dl0_empty(waveform) is False

    calibrator = CameraCalibrator(subarray=subarray)
    event = DataContainer()
    event.dl1.tel[telid].image = np.full(2048, 2)
    with pytest.warns(UserWarning):
        calibrator(event)
    assert (event.dl1.tel[telid].image == 2).all()


def test_dl1_charge_calib():
    camera = CameraGeometry.from_name("CHEC")
    n_pixels = camera.n_pixels
    n_samples = 96
    mid = n_samples // 2
    pulse_sigma = 6
    random = np.random.RandomState(1)
    x = np.arange(n_samples)

    # Randomize times and create pulses
    time_offset = random.uniform(mid - 10, mid + 10, n_pixels)[:, np.newaxis]
    y = norm.pdf(x, time_offset, pulse_sigma)

    # Define absolute calibration coefficients
    absolute = random.uniform(100, 1000, n_pixels)
    y *= absolute[:, np.newaxis]

    # Define relative coefficients
    relative = random.normal(1, 0.01, n_pixels)
    y /= relative[:, np.newaxis]

    # Define pedestal
    pedestal = random.uniform(-4, 4, n_pixels)
    y += pedestal[:, np.newaxis]

    event = DataContainer()
    telid = 0
    event.dl0.tel[telid].waveform = y

    # Test default
    calibrator = CameraCalibrator(
        subarray=subarray,
        image_extractor=FullWaveformSum(subarray=subarray)
    )
    calibrator(event)
    np.testing.assert_allclose(event.dl1.tel[telid].image, y.sum(1))

    event.calibration.tel[telid].dl1.time_shift = time_offset
    event.calibration.tel[telid].dl1.pedestal_offset = pedestal * n_samples
    event.calibration.tel[telid].dl1.absolute_factor = absolute
    event.calibration.tel[telid].dl1.relative_factor = relative

    # Test without need for timing corrections
    calibrator = CameraCalibrator(
        subarray=subarray,
        image_extractor=FullWaveformSum(subarray=subarray)
    )
    calibrator(event)
    np.testing.assert_allclose(event.dl1.tel[telid].image, 1)

    # TODO: Test with timing corrections
