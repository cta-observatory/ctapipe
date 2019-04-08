from numpy.testing import assert_array_equal

from ctapipe.calib.camera.r1 import (
    CameraR1Calibrator,
    HESSIOR1Calibrator,
    NullR1Calibrator,
)
from ctapipe.io.eventsource import EventSource
from ctapipe.io.simteleventsource import SimTelEventSource
from ctapipe.utils import get_dataset_path


dataset = get_dataset_path("gamma_test_large.simtel.gz")


def test_hessio_r1_calibrator(example_event):
    telid = list(example_event.r0.tel)[0]

    calibrator = HESSIOR1Calibrator()
    calibrator.calibrate(example_event)
    assert example_event.r1.tel[telid].waveform is not None


def test_null_r1_calibrator(example_event):
    telid = list(example_event.r0.tel)[0]

    calibrator = NullR1Calibrator()
    calibrator.calibrate(example_event)
    r0 = example_event.r0.tel[telid].waveform
    r1 = example_event.r1.tel[telid].waveform
    assert_array_equal(r0, r1)


def test_check_r0_exists(example_event):
    telid = list(example_event.r0.tel)[0]

    calibrator = HESSIOR1Calibrator()
    assert (calibrator.check_r0_exists(example_event, telid) is True)
    example_event.r0.tel[telid].waveform = None
    assert (calibrator.check_r0_exists(example_event, telid) is False)


def test_factory_from_product():
    calibrator = CameraR1Calibrator.from_name("NullR1Calibrator")
    assert isinstance(calibrator, NullR1Calibrator)
    calibrator = CameraR1Calibrator.from_name("HESSIOR1Calibrator")
    assert isinstance(calibrator, HESSIOR1Calibrator)


def test_factory_for_eventsource():
    eventsource = SimTelEventSource(input_url=dataset)
    calibrator = CameraR1Calibrator.from_eventsource(eventsource=eventsource)
    assert isinstance(calibrator, HESSIOR1Calibrator)


class UnknownEventSource(EventSource):
    """
    Simple working EventSource
    """

    def _generator(self):
        return range(len(self.input_url))

    @staticmethod
    def is_compatible(file_path):
        return False


def test_factory_from_unknown_eventsource():
    eventsource = UnknownEventSource(input_url=dataset)
    calibrator = CameraR1Calibrator.from_eventsource(eventsource=eventsource)
    assert isinstance(calibrator, NullR1Calibrator)
