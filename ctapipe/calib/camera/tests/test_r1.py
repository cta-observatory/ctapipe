from numpy.testing import assert_almost_equal
from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils import get_dataset
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory, \
    HessioR1Calibrator
from copy import deepcopy

def test_hessio_r1_calibrator(test_event):
    telid = 11
    event = deepcopy(test_event)
    calibrator = HessioR1Calibrator(None, None)
    calibrator.calibrate(event)
    r1 = event.r1.tel[telid].pe_samples
    assert_almost_equal(r1[0, 0, 0], -0.091, 3)


def test_check_r0_exists(test_event):
    telid = 11
    event = deepcopy(test_event)
    calibrator = HessioR1Calibrator(None, None)
    assert(calibrator.check_r0_exists(event, telid) is True)
    event.r0.tel[telid].adc_samples = None
    assert(calibrator.check_r0_exists(event, telid) is False)


def test_factory(test_event):
    calibrator = CameraR1CalibratorFactory.produce(None, None)

    telid = 11
    event = deepcopy(test_event)
    calibrator.calibrate(event)
    r1 = event.r1.tel[telid].pe_samples
    assert_almost_equal(r1[0, 0, 0], -0.091, 3)
