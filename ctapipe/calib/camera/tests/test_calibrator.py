from ctapipe.calib.camera import (
    CameraCalibrator,
)
from ctapipe.image.extractor import LocalPeakWindowSum
from traitlets.config.configurable import Config


def test_camera_calibrator(example_event):
    telid = list(example_event.r0.tel)[0]

    calibrator = CameraCalibrator()

    calibrator.calibrate(example_event)
    image = example_event.dl1.tel[telid].image
    assert image is not None


def test_manual_extractor():
    calibrator = CameraCalibrator(extractor_name="LocalPeakWindowSum")
    assert isinstance(calibrator.dl1.extractor, LocalPeakWindowSum)


def test_config():
    window_shift = 3
    window_width = 9
    config = Config({"LocalPeakWindowSum": {
        "window_shift": window_shift,
        "window_width": window_width,
    }})
    calibrator = CameraCalibrator(
        extractor_name='LocalPeakWindowSum',
        config=config
    )
    assert calibrator.dl1.extractor.window_shift == window_shift
    assert calibrator.dl1.extractor.window_width == window_width
