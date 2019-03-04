import copy
import pytest
from ctapipe.io.targetioeventsource import TargetIOEventSource
from ctapipe.io import event_source
from ctapipe.io.eventseeker import EventSeeker
from ctapipe.utils import get_dataset_path
from ctapipe.calib.camera.calibrator import CameraCalibrator

pytest.importorskip("target_driver")
pytest.importorskip("target_io")
pytest.importorskip("target_calib")


def test_chec_r1():
    url = get_dataset_path("chec_r1.tio")
    source = TargetIOEventSource(input_url=url)
    event = source._get_event_by_index(0)
    assert(source._r0_samples is None)
    assert(source._r1_samples.shape[1] == 2048)
    assert(round(source._r1_samples[0, 0, 0]) == -274)
    assert(event.r0.tels_with_data == {0})
    assert(event.r0.tel[0].waveform is None)
    assert(event.r1.tel[0].waveform[0, 0, 0] == source._r1_samples[0, 0, 0])


def test_event_id():
    url = get_dataset_path("chec_r1.tio")
    source = TargetIOEventSource(input_url=url)
    event_id = 2
    source._get_event_by_id(event_id)
    assert(event_id == source._reader.fCurrentEventID)
    assert(round(source._r1_samples[0, 0, 0]) == -274)


def test_singlemodule_r0():
    url = get_dataset_path("targetmodule_r0.tio")
    source = TargetIOEventSource(input_url=url)
    event = source._get_event_by_index(0)
    assert(round(source._r0_samples[0, 0, 0]) == 600)
    assert(source._r1_samples is None)
    assert(event.r0.tels_with_data == {0})
    assert(event.r0.tel[0].waveform[0, 0, 0] == source._r0_samples[0, 0, 0])


def test_singlemodule_r1():
    url = get_dataset_path("targetmodule_r1.tio")
    source = TargetIOEventSource(input_url=url)
    event = source._get_event_by_index(0)
    assert(source._r0_samples is None)
    assert(source._r1_samples.shape[1] == 64)
    assert(round(source._r1_samples[0, 0, 0]) == 0)
    assert(event.r0.tels_with_data == {0})
    assert(event.r0.tel[0].waveform is None)
    assert(event.r1.tel[0].waveform[0, 0, 0] == source._r1_samples[0, 0, 0])


def test_compatible():
    dataset = get_dataset_path("chec_r1.tio")
    assert TargetIOEventSource.is_compatible(dataset)

    dataset = get_dataset_path("gamma_test_large.simtel.gz")
    assert not TargetIOEventSource.is_compatible(dataset)


def test_stream():
    dataset = get_dataset_path("chec_r1.tio")
    with TargetIOEventSource(input_url=dataset) as source:
        assert not source.is_stream


def test_loop():
    dataset = get_dataset_path("chec_r1.tio")
    with TargetIOEventSource(input_url=dataset) as source:
        count = 0
        for event in source:
            assert event.r0.tels_with_data == {0}
            assert event.count == count
            count += 1

        for event in source:
            # Check generator has restarted from beginning
            assert event.count == 0
            break


def test_that_event_is_not_modified_after_loop():
    dataset = get_dataset_path("chec_r1.tio")
    with TargetIOEventSource(input_url=dataset, max_events=2) as source:
        for event in source:
            last_event = copy.deepcopy(event)

        # now `event` should be identical with the deepcopy of itself from
        # inside the loop.
        # Unfortunately this does not work:
        #      assert last_event == event
        # So for the moment we just compare event ids
        assert event.r0.event_id == last_event.r0.event_id


def test_len():
    dataset = get_dataset_path("chec_r1.tio")
    with TargetIOEventSource(input_url=dataset) as source:
        count = 0
        for _ in source:
            count += 1
        assert count == len(source)

    with TargetIOEventSource(input_url=dataset, max_events=3) as reader:
        assert len(reader) == 3


def test_geom():
    dataset = get_dataset_path("chec_r1.tio")
    with TargetIOEventSource(input_url=dataset) as source:
        event = source._get_event_by_index(0)
        assert event.inst.subarray.tels[0].camera.pix_x.size == 2048

    dataset = get_dataset_path("targetmodule_r1.tio")
    with TargetIOEventSource(input_url=dataset) as source:
        event = source._get_event_by_index(0)
        assert event.inst.subarray.tels[0].camera.pix_x.size == 64


def test_eventsourcefactory():
    dataset = get_dataset_path("chec_r1.tio")
    source = event_source(dataset)
    assert source.__class__.__name__ == "TargetIOEventSource"
    assert source.input_url == dataset


def test_eventseeker():
    dataset = get_dataset_path("chec_r1.tio")
    with TargetIOEventSource(input_url=dataset) as source:
        seeker = EventSeeker(source)
        event = seeker[0]
        assert source._event_index == 0
        assert source._event_id == 2
        assert event.count == 0
        assert event.r1.event_id == 2
        assert (round(source._r1_samples[0, 0, 0]) == -274)

        event = seeker['2']
        assert source._event_index == 0
        assert source._event_id == 2
        assert event.count == 0
        assert event.r1.event_id == 2
        assert (round(source._r1_samples[0, 0, 0]) == -274)

        event = seeker[-1]
        assert event.count == len(seeker) - 1

    with TargetIOEventSource(input_url=dataset, max_events=3) as source:
        with pytest.raises(IndexError):
            seeker = EventSeeker(source)
            seeker[5]


def test_pipeline():
    dataset = get_dataset_path("chec_r1.tio")
    reader = TargetIOEventSource(input_url=dataset, max_events=10)
    calibrator = CameraCalibrator(eventsource=reader)
    for event in reader:
        calibrator.calibrate(event)
        assert event.r0.tel.keys() == event.dl1.tel.keys()
