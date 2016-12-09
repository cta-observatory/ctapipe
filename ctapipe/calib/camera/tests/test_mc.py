from numpy.testing import assert_almost_equal
from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils.datasets import get_path
from ctapipe.calib.camera.mc import mc_r0_to_dl0_calibration
from IPython import embed


def get_test_event():
    filename = get_path('gamma_test.simtel.gz')
    for event in hessio_event_source(filename):
        if event.dl0.event_id == 409:
            return event


def test_mc_r0_to_dl0_calibration():
    telid = 11
    event = get_test_event()
    dl0 = mc_r0_to_dl0_calibration(event, telid)
    assert_almost_equal(dl0[0, 0, 0].value, -0.087, 3)
