from numpy.testing import assert_almost_equal, assert_array_equal
from ctapipe.calib.camera.r1 import (
    CameraR1CalibratorFactory,
    HessioR1Calibrator,
    NullR1Calibrator
)
from copy import deepcopy

def test_hessio_r1_calibrator(test_event):
    telid = 11
    event = deepcopy(test_event)
    calibrator = HessioR1Calibrator()
    calibrator.calibrate(event)
    r1 = event.r1.tel[telid].pe_samples
    assert_almost_equal(r1[0, 0, 0], -0.091, 3)


def test_null_r1_calibrator(test_event):
    telid = 11
    event = deepcopy(test_event)
    calibrator = NullR1Calibrator()
    calibrator.calibrate(event)
    r0 = event.r0.tel[telid].adc_samples
    r1 = event.r1.tel[telid].pe_samples
    assert_array_equal(r0, r1)


def test_check_r0_exists(test_event):
    telid = 11
    event = deepcopy(test_event)
    calibrator = HessioR1Calibrator()
    assert(calibrator.check_r0_exists(event, telid) is True)
    event.r0.tel[telid].adc_samples = None
    assert(calibrator.check_r0_exists(event, telid) is False)


def test_factory(test_event):
    calibrator = CameraR1CalibratorFactory.produce()

    telid = 11
    event = deepcopy(test_event)
    calibrator.calibrate(event)
    r1 = event.r1.tel[telid].pe_samples
    assert_almost_equal(r1[0, 0, 0], -0.091, 3)
