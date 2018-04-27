from copy import deepcopy

from numpy.testing import assert_almost_equal

from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.r1 import HESSIOR1Calibrator


def previous_calibration(event):
    r1 = HESSIOR1Calibrator()
    r1.calibrate(event)


def test_camera_dl0_reducer(test_event):
    event = deepcopy(test_event)
    previous_calibration(event)
    telid = 11

    reducer = CameraDL0Reducer()
    reducer.reduce(event)
    waveforms = event.dl0.tel[telid].waveform
    assert_almost_equal(waveforms[0, 0, 0], -0.091, 3)


def test_check_r1_exists(test_event):
    event = deepcopy(test_event)
    telid = 11
    previous_calibration(event)
    reducer = CameraDL0Reducer()
    assert(reducer.check_r1_exists(event, telid) is True)
    event.r1.tel[telid].waveform = None
    assert(reducer.check_r1_exists(event, telid) is False)
