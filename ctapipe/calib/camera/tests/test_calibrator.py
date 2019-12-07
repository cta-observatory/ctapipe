from ctapipe.calib.camera.calibrator import (
    CameraCalibrator,
    integration_correction,
)
from ctapipe.instrument import CameraGeometry
from ctapipe.io.containers import DataContainer, EventAndMonDataContainer
from ctapipe.image.extractor import LocalPeakWindowSum, FullWaveformSum
from traitlets.config.configurable import Config
import pytest
import numpy as np
from scipy.stats import norm


def test_camera_calibrator(example_event):
    telid = list(example_event.r0.tel)[0]
    calibrator = CameraCalibrator(subarray=example_event.inst.subarray)
    calibrator(example_event)
    image = example_event.dl1.tel[telid].image
    pulse_time = example_event.dl1.tel[telid].pulse_time
    assert image is not None
    assert pulse_time is not None
    assert image.shape == (1764,)
    assert pulse_time.shape == (1764,)


def test_manual_extractor():
    calibrator = CameraCalibrator(image_extractor=LocalPeakWindowSum())
    assert isinstance(calibrator.image_extractor, LocalPeakWindowSum)


def test_config():
    window_shift = 3
    window_width = 9
    config = Config({"LocalPeakWindowSum": {
        "window_shift": window_shift,
        "window_width": window_width,
    }})
    calibrator = CameraCalibrator(
        image_extractor=LocalPeakWindowSum(config=config),
        config=config
    )
    assert calibrator.image_extractor.window_shift[None] == window_shift
    assert calibrator.image_extractor.window_width[None] == window_width


def test_integration_correction(example_event):
    telid = list(example_event.r0.tel)[0]

    width = 7
    shift = 3
    shape = example_event.mc.tel[telid].reference_pulse_shape
    n_chan = shape.shape[0]
    step = example_event.mc.tel[telid].meta['refstep']
    time_slice = example_event.mc.tel[telid].time_slice
    correction = integration_correction(n_chan, shape, step,
                                        time_slice, width, shift)
    assert correction is not None


def test_integration_correction_no_ref_pulse(example_event):
    telid = list(example_event.r0.tel)[0]
    delattr(example_event, 'mc')
    calibrator = CameraCalibrator()
    calibrator._calibrate_dl0(example_event, telid)
    correction = calibrator._get_correction(example_event, telid)
    assert (correction == 1).all()


def test_check_r1_empty(example_event):
    calibrator = CameraCalibrator()
    telid = list(example_event.r0.tel)[0]
    waveform = example_event.r1.tel[telid].waveform.copy()
    with pytest.warns(UserWarning):
        example_event.r1.tel[telid].waveform = None
        calibrator._calibrate_dl0(example_event, telid)
        assert example_event.dl0.tel[telid].waveform is None

    assert calibrator._check_r1_empty(None) is True
    assert calibrator._check_r1_empty(waveform) is False

    calibrator = CameraCalibrator(image_extractor=FullWaveformSum())
    event = DataContainer()
    event.dl0.tel[telid].waveform = np.full((2048, 128), 2)
    with pytest.warns(UserWarning):
        calibrator(event)
    assert (event.dl0.tel[telid].waveform == 2).all()
    assert (event.dl1.tel[telid].image == 2*128).all()


def test_check_dl0_empty(example_event):
    calibrator = CameraCalibrator()
    telid = list(example_event.r0.tel)[0]
    calibrator._calibrate_dl0(example_event, telid)
    waveform = example_event.dl0.tel[telid].waveform.copy()
    with pytest.warns(UserWarning):
        example_event.dl0.tel[telid].waveform = None
        calibrator._calibrate_dl1(example_event, telid)
        assert example_event.dl1.tel[telid].image is None

    assert calibrator._check_dl0_empty(None) is True
    assert calibrator._check_dl0_empty(waveform) is False

    calibrator = CameraCalibrator()
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
    calibrator = CameraCalibrator(image_extractor=FullWaveformSum())
    calibrator(event)
    np.testing.assert_allclose(event.dl1.tel[telid].image, y.sum(1))

    event.calibration.tel[telid].dl1.time_shift = time_offset
    event.calibration.tel[telid].dl1.pedestal_offset = pedestal * n_samples
    event.calibration.tel[telid].dl1.absolute_factor = absolute
    event.calibration.tel[telid].dl1.relative_factor = relative

    # Test without need for timing corrections
    calibrator = CameraCalibrator(image_extractor=FullWaveformSum())
    calibrator(event)
    np.testing.assert_allclose(event.dl1.tel[telid].image, 1)

    # TODO: Test with timing corrections
