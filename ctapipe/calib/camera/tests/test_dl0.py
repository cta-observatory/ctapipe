from numpy.testing import assert_almost_equal

from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.r1 import HESSIOR1Calibrator


def previous_calibration(event):
    r1 = HESSIOR1Calibrator()
    r1.calibrate(event)


def test_camera_dl0_reducer(example_event):
    previous_calibration(example_event)
    telid = 11

    reducer = CameraDL0Reducer()
    reducer.reduce(example_event)
    waveforms = example_event.dl0.tel[telid].waveform
    assert_almost_equal(waveforms[0, 0, 0], -0.091, 3)


def test_check_r1_exists(example_event):
    telid = 11
    previous_calibration(example_event)
    reducer = CameraDL0Reducer()
    assert (reducer.check_r1_exists(example_event, telid) is True)
    example_event.r1.tel[telid].waveform = None
    assert (reducer.check_r1_exists(example_event, telid) is False)
