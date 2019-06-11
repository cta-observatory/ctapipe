from ctapipe.calib.camera.calibrator import (
    CameraCalibrator,
    integration_correction,
)
from ctapipe.image.extractor import LocalPeakWindowSum
from traitlets.config.configurable import Config
import pytest


def test_camera_calibrator(example_event):
    telid = list(example_event.r0.tel)[0]
    calibrator = CameraCalibrator()
    calibrator(example_event)
    image = example_event.dl1.tel[telid].image
    assert image is not None


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
    assert calibrator.image_extractor.window_shift == window_shift
    assert calibrator.image_extractor.window_width == window_width


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
    assert correction[0] == 1


def test_check_r1_empty(example_event):
    calibrator = CameraCalibrator()
    telid = list(example_event.r0.tel)[0]
    waveform = example_event.r1.tel[telid].waveform.copy()
    with pytest.warns(UserWarning):
        example_event.r1.tel[telid].waveform = None
        calibrator._calibrate_dl0(example_event, telid)
        assert example_event.dl0.tel[telid].waveform == None

    assert calibrator._check_r1_empty(None) is True
    assert calibrator._check_r1_empty(waveform) is False


def test_check_dl0_empty(example_event):
    calibrator = CameraCalibrator()
    telid = list(example_event.r0.tel)[0]
    calibrator._calibrate_dl0(example_event, telid)
    waveform = example_event.dl0.tel[telid].waveform.copy()
    with pytest.warns(UserWarning):
        example_event.dl0.tel[telid].waveform = None
        calibrator._calibrate_dl1(example_event, telid)
        assert example_event.dl1.tel[telid].image == None

    assert calibrator._check_dl0_empty(None) is True
    assert calibrator._check_dl0_empty(waveform) is False
