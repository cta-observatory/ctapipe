from numpy.testing import assert_almost_equal
from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils.datasets import get_path
from ctapipe.calib.camera.r1 import CameraR1Calibrator, HessioR1Calibrator


def get_test_event():
    filename = get_path('gamma_test.simtel.gz')
    source = hessio_event_source(filename, requested_event=409,
                                 use_event_id=True)
    event = next(source)
    return event


def test_mc_r0_to_dl0_calibration():
    telid = 11
    event = get_test_event()
    dl0 = mc_r0_to_dl0_calibration(event, telid)
    assert_almost_equal(dl0[0, 0, 0], -0.091, 3)
