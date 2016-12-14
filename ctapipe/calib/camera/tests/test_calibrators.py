from ctapipe.calib.camera.dl1 import integration_correction, \
    CameraDL1Calibrator
from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils.datasets import get_path
from numpy.testing import assert_almost_equal


def get_test_event():
    filename = get_path('gamma_test.simtel.gz')
    source = hessio_event_source(filename, requested_event=409,
                                 use_event_id=True)
    event = next(source)
    return event


def test_integration_correction():
    event = get_test_event()
    telid = 11
    width = 7
    shift = 3
    correction = integration_correction(event, telid, width, shift)
    assert_almost_equal(correction[0], 2.15, 2)


def test_camera_dl1_calibrator():
    event = get_test_event()
    telid = 11
    calibrator = CameraDL1Calibrator(None, None)

    calibrator.get_neighbours(event, telid)
    assert(calibrator.neighbour_dict[telid][0][0] == 5)

    calibrator.get_correction(event, telid)
    assert_almost_equal(calibrator.correction_dict[telid][0], 2.154, 3)

    dl0 = calibrator.obtain_dl0(event, telid)
    assert_almost_equal(dl0[0, 0, 0], -0.091, 3)

    calibrator.calibrate(event)
    image = event.dl1.tel[telid].image
    assert_almost_equal(image[0, 0], -4.431, 3)

    filename = get_path('gamma_test.simtel.gz')
    source = hessio_event_source(filename, requested_event=409,
                                 use_event_id=True)
    calibrator.calibrate_source(source)
    event = next(source)
    assert_almost_equal(event.dl1.tel[21].image[0, 0], -3.410, 3)
