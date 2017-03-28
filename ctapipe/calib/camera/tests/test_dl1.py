from ctapipe.calib.camera.dl1 import integration_correction, \
    CameraDL1Calibrator
from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.r1 import HessioR1Calibrator
from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils.datasets import get_path
from numpy.testing import assert_almost_equal


def get_test_event():
    filename = get_path('gamma_test.simtel.gz')
    source = hessio_event_source(filename, requested_event=409,
                                 use_event_id=True)
    event = next(source)
    return event


def previous_calibration(event):
    r1 = HessioR1Calibrator(None, None)
    r1.calibrate(event)
    dl0 = CameraDL0Reducer(None, None)
    dl0.reduce(event)


def test_integration_correction():
    event = get_test_event()
    telid = 11
    width = 7
    shift = 3
    correction = integration_correction(event, telid, width, shift)
    assert_almost_equal(correction[0], 2.15, 2)


def test_camera_dl1_calibrator():
    event = get_test_event()
    previous_calibration(event)
    telid = 11

    calibrator = CameraDL1Calibrator(None, None)

    calibrator.get_neighbours(event, telid)
    assert(calibrator.neighbour_dict[telid][0][0] == 5)

    calibrator.get_correction(event, telid)
    assert_almost_equal(calibrator.correction_dict[telid][0], 2.154, 3)

    calibrator.calibrate(event)
    image = event.dl1.tel[telid].image
    assert_almost_equal(image[0, 0], -4.431, 3)


def test_check_dl0_exists():
    telid = 11
    event = get_test_event()
    previous_calibration(event)
    calibrator = CameraDL1Calibrator(None, None)
    assert(calibrator.check_dl0_exists(event, telid) is True)
    event.dl0.tel[telid].pe_samples = None
    assert(calibrator.check_dl0_exists(event, telid) is False)
