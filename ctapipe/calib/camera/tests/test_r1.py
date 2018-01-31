from numpy.testing import assert_almost_equal, assert_array_equal
from ctapipe.calib.camera.r1 import (
    CameraR1CalibratorFactory,
    HESSIOR1Calibrator,
    NullR1Calibrator
)
from ctapipe.io.hessioeventsource import HESSIOEventSource
from ctapipe.io.eventsource import EventSource
from ctapipe.utils import get_dataset
from copy import deepcopy


def test_hessio_r1_calibrator(test_event):
    telid = 11
    event = deepcopy(test_event)
    calibrator = HESSIOR1Calibrator()
    calibrator.calibrate(event)
    r1 = event.r1.tel[telid].pe_samples
    assert_almost_equal(r1[0, 0, 0], -0.091, 3)


def test_null_r1_calibrator(test_event):
    telid = 11
    event = deepcopy(test_event)
    calibrator = NullR1Calibrator()
    calibrator.calibrate(event)
    r0 = event.r0.tel[telid].adc_samples
    r1 = event.r1.tel[telid].pe_samples
    assert_array_equal(r0, r1)


def test_check_r0_exists(test_event):
    telid = 11
    event = deepcopy(test_event)
    calibrator = HESSIOR1Calibrator()
    assert(calibrator.check_r0_exists(event, telid) is True)
    event.r0.tel[telid].adc_samples = None
    assert(calibrator.check_r0_exists(event, telid) is False)


def test_factory_from_product():
    calibrator = CameraR1CalibratorFactory.produce(
        product="NullR1Calibrator"
    )
    assert isinstance(calibrator, NullR1Calibrator)
    calibrator = CameraR1CalibratorFactory.produce(
        product="HESSIOR1Calibrator"
    )
    assert isinstance(calibrator, HESSIOR1Calibrator)


def test_factory_default():
    calibrator = CameraR1CalibratorFactory.produce()
    assert isinstance(calibrator, NullR1Calibrator)


def test_factory_from_eventsource():
    dataset = get_dataset("gamma_test.simtel.gz")
    eventsource = HESSIOEventSource(input_url=dataset)
    calibrator = CameraR1CalibratorFactory.produce(eventsource=eventsource)
    assert isinstance(calibrator, HESSIOR1Calibrator)


def test_factory_from_eventsource_override():
    dataset = get_dataset("gamma_test.simtel.gz")
    eventsource = HESSIOEventSource(input_url=dataset)
    calibrator = CameraR1CalibratorFactory.produce(
        eventsource=eventsource,
        product="NullR1Calibrator"
    )
    assert isinstance(calibrator, NullR1Calibrator)


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
    dataset = get_dataset("gamma_test.simtel.gz")
    eventsource = UnknownEventSource(input_url=dataset)
    calibrator = CameraR1CalibratorFactory.produce(eventsource=eventsource)
    assert isinstance(calibrator, NullR1Calibrator)
