from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.r1 import HessioR1Calibrator
from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils import get_dataset
from numpy.testing import assert_almost_equal


def get_test_event():
    filename = get_dataset('gamma_test.simtel.gz')
    source = hessio_event_source(filename, requested_event=409,
                                 use_event_id=True)
    event = next(source)
    return event


def previous_calibration(event):
    r1 = HessioR1Calibrator(None, None)
    r1.calibrate(event)


def test_camera_dl0_reducer():
    event = get_test_event()
    previous_calibration(event)
    telid = 11

    reducer = CameraDL0Reducer(None, None)
    reducer.reduce(event)
    waveforms = event.dl0.tel[telid].pe_samples
    assert_almost_equal(waveforms[0, 0, 0], -0.091, 3)


def test_check_r1_exists():
    telid = 11
    event = get_test_event()
    previous_calibration(event)
    reducer = CameraDL0Reducer(None, None)
    assert(reducer.check_r1_exists(event, telid) is True)
    event.r1.tel[telid].pe_samples = None
    assert(reducer.check_r1_exists(event, telid) is False)
