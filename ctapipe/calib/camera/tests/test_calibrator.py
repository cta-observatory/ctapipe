from copy import deepcopy

from numpy.testing import assert_allclose

from ctapipe.calib.camera import CameraCalibrator


def test_camera_calibrator(test_event):
    event = deepcopy(test_event) # so we don't modify the test event
    telid = 11

    calibrator = CameraCalibrator()

    calibrator.calibrate(event)
    image = event.dl1.tel[telid].image
    assert_allclose(image[0, 0], -2.216, 1e-3)
