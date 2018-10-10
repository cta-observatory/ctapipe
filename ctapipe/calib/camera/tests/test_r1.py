import pytest
from numpy.testing import assert_almost_equal, assert_array_equal, \
    assert_array_almost_equal

from ctapipe.calib.camera.r1 import (
    CameraR1CalibratorFactory,
    HESSIOR1Calibrator,
    TargetIOR1Calibrator,
    NullR1Calibrator
)
from ctapipe.io.eventsource import EventSource
from ctapipe.io.hessioeventsource import HESSIOEventSource
from ctapipe.io.targetioeventsource import TargetIOEventSource
from ctapipe.utils import get_dataset_path


def test_hessio_r1_calibrator(example_event):
    telid = 11

    calibrator = HESSIOR1Calibrator()
    calibrator.calibrate(example_event)
    r1 = example_event.r1.tel[telid].waveform
    assert_almost_equal(r1[0, 0, 0], -0.091, 3)


def test_null_r1_calibrator(example_event):
    telid = 11

    calibrator = NullR1Calibrator()
    calibrator.calibrate(example_event)
    r0 = example_event.r0.tel[telid].waveform
    r1 = example_event.r1.tel[telid].waveform
    assert_array_equal(r0, r1)


def test_targetio_calibrator():
    pytest.importorskip("target_calib")
    url_r0 = get_dataset_path("targetmodule_r0.tio")
    url_r1 = get_dataset_path("targetmodule_r1.tio")
    pedpath = get_dataset_path("targetmodule_ped.tcal")

    source_r0 = TargetIOEventSource(input_url=url_r0)
    source_r1 = TargetIOEventSource(input_url=url_r1)

    r1c = CameraR1CalibratorFactory.produce(eventsource=source_r0)

    event_r0 = source_r0._get_event_by_index(0)
    event_r1 = source_r1._get_event_by_index(0)

    r1c.calibrate(event_r0)
    assert_array_equal(event_r0.r0.tel[0].waveform,
                       event_r0.r1.tel[0].waveform)

    r1c = CameraR1CalibratorFactory.produce(
        eventsource=source_r0,
        pedestal_path=pedpath
    )
    r1c.calibrate(event_r0)
    assert_array_almost_equal(event_r0.r1.tel[0].waveform,
                              event_r1.r1.tel[0].waveform, 1)


def test_targetio_calibrator_wrong_file(example_event):
    pytest.importorskip("target_calib")
    r1c = TargetIOR1Calibrator()
    with pytest.raises(ValueError):
        r1c.calibrate(example_event)


def test_check_r0_exists(example_event):
    telid = 11

    calibrator = HESSIOR1Calibrator()
    assert (calibrator.check_r0_exists(example_event, telid) is True)
    example_event.r0.tel[telid].waveform = None
    assert (calibrator.check_r0_exists(example_event, telid) is False)


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
    dataset = get_dataset_path("gamma_test.simtel.gz")
    eventsource = HESSIOEventSource(input_url=dataset)
    calibrator = CameraR1CalibratorFactory.produce(eventsource=eventsource)
    assert isinstance(calibrator, HESSIOR1Calibrator)


def test_factory_from_eventsource_override():
    dataset = get_dataset_path("gamma_test.simtel.gz")
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
    dataset = get_dataset_path("gamma_test.simtel.gz")
    eventsource = UnknownEventSource(input_url=dataset)
    calibrator = CameraR1CalibratorFactory.produce(eventsource=eventsource)
    assert isinstance(calibrator, NullR1Calibrator)
