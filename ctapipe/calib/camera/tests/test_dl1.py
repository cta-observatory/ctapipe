from copy import deepcopy

from numpy.testing import assert_allclose

from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import integration_correction, \
    CameraDL1Calibrator
from ctapipe.calib.camera.r1 import HESSIOR1Calibrator


def previous_calibration(event):
    r1 = HESSIOR1Calibrator()
    r1.calibrate(event)
    dl0 = CameraDL0Reducer()
    dl0.reduce(event)


def test_integration_correction(test_event):
    event = deepcopy(test_event)
    telid = 11
    width = 7
    shift = 3
    shape = event.mc.tel[telid].reference_pulse_shape
    n_chan = shape.shape[0]
    step = event.mc.tel[telid].meta['refstep']
    time_slice = event.mc.tel[telid].time_slice
    correction = integration_correction(n_chan, shape, step,
                                        time_slice, width, shift)
    assert_allclose(correction[0], 1.077, 1e-3)


def test_camera_dl1_calibrator(test_event):
    event = deepcopy(test_event)
    previous_calibration(event)
    telid = 11

    calibrator = CameraDL1Calibrator()

    correction = calibrator.get_correction(event, telid)
    assert_allclose(correction[0], 1.077, 1e-3)

    calibrator.calibrate(event)
    image = event.dl1.tel[telid].image
    assert_allclose(image[0, 0], -2.216, 1e-3)


def test_check_dl0_exists(test_event):
    event = deepcopy(test_event)
    telid = 11
    previous_calibration(event)
    calibrator = CameraDL1Calibrator()
    assert(calibrator.check_dl0_exists(event, telid) is True)
    event.dl0.tel[telid].waveform = None
    assert(calibrator.check_dl0_exists(event, telid) is False)
