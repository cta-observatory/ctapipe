from numpy.testing import assert_almost_equal
from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils import get_dataset
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory, \
    HessioR1Calibrator


def get_test_event():
    filename = get_dataset('gamma_test.simtel.gz')
    source = hessio_event_source(filename, requested_event=409,
                                 use_event_id=True)
    event = next(source)
    return event


def test_hessio_r1_calibrator():
    telid = 11
    event = get_test_event()
    calibrator = HessioR1Calibrator(None, None)
    calibrator.calibrate(event)
    r1 = event.r1.tel[telid].pe_samples
    assert_almost_equal(r1[0, 0, 0], -0.091, 3)


def test_check_r0_exists():
    telid = 11
    event = get_test_event()
    calibrator = HessioR1Calibrator(None, None)
    assert(calibrator.check_r0_exists(event, telid) is True)
    event.r0.tel[telid].adc_samples = None
    assert(calibrator.check_r0_exists(event, telid) is False)


def test_factory():
    factory = CameraR1CalibratorFactory(None, None)
    cls = factory.get_class()
    calibrator = cls(None, None)

    telid = 11
    event = get_test_event()
    calibrator.calibrate(event)
    r1 = event.r1.tel[telid].pe_samples
    assert_almost_equal(r1[0, 0, 0], -0.091, 3)
