from numpy.testing import assert_allclose

from ctapipe.calib.camera import (
    CameraCalibrator,
    HESSIOR1Calibrator,
    NullR1Calibrator
)
from ctapipe.image.extractor import LocalPeakWindowSum
from ctapipe.io import SimTelEventSource
from ctapipe.utils import get_dataset_path
from traitlets.config.configurable import Config


def test_camera_calibrator(example_event):
    telid = list(example_event.r0.tel)[0]

    calibrator = CameraCalibrator(r1_product="HESSIOR1Calibrator")

    calibrator.calibrate(example_event)
    image = example_event.dl1.tel[telid].image
    assert image is not None


def test_manual_r1():
    calibrator = CameraCalibrator(r1_product="HESSIOR1Calibrator")
    assert isinstance(calibrator.r1, HESSIOR1Calibrator)


def test_manual_extractor():
    calibrator = CameraCalibrator(extractor_name="LocalPeakWindowSum")
    assert isinstance(calibrator.dl1.extractor, LocalPeakWindowSum)


def test_eventsource_r1():
    dataset = get_dataset_path("gamma_test_large.simtel.gz")
    eventsource = SimTelEventSource(input_url=dataset)
    calibrator = CameraCalibrator(eventsource=eventsource)
    assert isinstance(calibrator.r1, HESSIOR1Calibrator)


def test_eventsource_override_r1():
    dataset = get_dataset_path("gamma_test_large.simtel.gz")
    eventsource = SimTelEventSource(input_url=dataset)
    calibrator = CameraCalibrator(
        eventsource=eventsource,
        r1_product="NullR1Calibrator"
    )
    assert isinstance(calibrator.r1, NullR1Calibrator)


def test_config():
    window_shift = 3
    window_width = 9
    config = Config({"LocalPeakWindowSum": {
        "window_shift": window_shift,
        "window_width": window_width,
    }})
    calibrator = CameraCalibrator(
        r1_product='HESSIOR1Calibrator',
        extractor_name='LocalPeakWindowSum',
        config=config
    )
    assert calibrator.dl1.extractor.window_shift == window_shift
    assert calibrator.dl1.extractor.window_width == window_width
