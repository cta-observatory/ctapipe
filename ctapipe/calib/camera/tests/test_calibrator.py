from copy import deepcopy
from numpy.testing import assert_allclose
from ctapipe.calib.camera import (
    CameraCalibrator,
    HESSIOR1Calibrator,
    NullR1Calibrator
)
from ctapipe.image.charge_extractors import LocalPeakIntegrator
from ctapipe.io import HESSIOEventSource
from ctapipe.utils import get_dataset_path


def test_camera_calibrator(test_event):
    event = deepcopy(test_event) # so we don't modify the test event
    telid = 11

    calibrator = CameraCalibrator(r1_product="HESSIOR1Calibrator")

    calibrator.calibrate(event)
    image = event.dl1.tel[telid].image
    assert_allclose(image[0, 0], -2.216, 1e-3)


def test_manual_r1():
    calibrator = CameraCalibrator(r1_product="HESSIOR1Calibrator")
    assert isinstance(calibrator.r1, HESSIOR1Calibrator)


def test_manual_extractor():
    calibrator = CameraCalibrator(extractor_product="LocalPeakIntegrator")
    assert isinstance(calibrator.dl1.extractor, LocalPeakIntegrator)


def test_eventsource_r1():
    dataset = get_dataset_path("gamma_test.simtel.gz")
    eventsource = HESSIOEventSource(input_url=dataset)
    calibrator = CameraCalibrator(eventsource=eventsource)
    assert isinstance(calibrator.r1, HESSIOR1Calibrator)


def test_eventsource_override_r1():
    dataset = get_dataset_path("gamma_test.simtel.gz")
    eventsource = HESSIOEventSource(input_url=dataset)
    calibrator = CameraCalibrator(
        eventsource=eventsource,
        r1_product="NullR1Calibrator"
    )
    assert isinstance(calibrator.r1, NullR1Calibrator)
