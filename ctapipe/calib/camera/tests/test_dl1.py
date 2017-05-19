from ctapipe.calib.camera.dl1 import integration_correction, \
    CameraDL1Calibrator
from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.r1 import HessioR1Calibrator
from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils import get_dataset
from numpy.testing import assert_allclose


def get_test_event():
    filename = get_dataset('gamma_test.simtel.gz')
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
    n_chan = event.inst.num_channels[telid]
    shape = event.mc.tel[telid].reference_pulse_shape
    step = event.mc.tel[telid].meta['refstep']
    time_slice = event.mc.tel[telid].time_slice
    correction = integration_correction(n_chan, shape, step,
                                        time_slice, width, shift)
    assert_allclose(correction[0], 1.077, 1e-3)


def test_camera_dl1_calibrator():
    event = get_test_event()
    previous_calibration(event)
    telid = 11

    calibrator = CameraDL1Calibrator(None, None)

    correction = calibrator.get_correction(event, telid)
    assert_allclose(correction[0], 1.077, 1e-3)

    calibrator.calibrate(event)
    image = event.dl1.tel[telid].image
    assert_allclose(image[0, 0], -2.216, 1e-3)


def test_check_dl0_exists():
    telid = 11
    event = get_test_event()
    previous_calibration(event)
    calibrator = CameraDL1Calibrator(None, None)
    assert(calibrator.check_dl0_exists(event, telid) is True)
    event.dl0.tel[telid].pe_samples = None
    assert(calibrator.check_dl0_exists(event, telid) is False)
