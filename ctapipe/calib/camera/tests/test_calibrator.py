from ctapipe.calib.camera import CameraCalibrator
from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils import get_dataset
from numpy.testing import assert_allclose


def get_test_event():
    filename = get_dataset('gamma_test.simtel.gz')
    source = hessio_event_source(filename, requested_event=409,
                                 use_event_id=True)
    event = next(source)
    return event


def test_camera_calibrator():
    event = get_test_event()
    telid = 11

    calibrator = CameraCalibrator(None, None)

    calibrator.calibrate(event)
    image = event.dl1.tel[telid].image
    assert_allclose(image[0, 0], -2.216, 1e-3)
