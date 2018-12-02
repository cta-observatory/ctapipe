import copy
import pytest
from ctapipe.io.magiceventsource import MAGICEventSourceROOT
from ctapipe.io.eventsourcefactory import EventSourceFactory
from ctapipe.io.eventseeker import EventSeeker
from ctapipe.utils import get_dataset_path


def test_compatible():
    m1_dataset = get_dataset_path("20131004_M1_05029747.003_Y_MagicCrab-W0.40+035.root")
    m2_dataset = get_dataset_path("20131004_M2_05029747.003_Y_MagicCrab-W0.40+035.root")
    assert MAGICEventSourceROOT.is_compatible(m1_dataset)
    assert MAGICEventSourceROOT.is_compatible(m2_dataset)


def test_stream():
    dataset = get_dataset_path("20131004_M1_05029747.003_Y_MagicCrab-W0.40+035.root")
    with MAGICEventSourceROOT(input_url=dataset) as source:
        assert not source.is_stream


def test_loop():
    dataset = get_dataset_path("20131004_M1_05029747.003_Y_MagicCrab-W0.40+035.root")
    dataset = dataset.replace('_M1_', '_M*_')
    with MAGICEventSourceROOT(input_url=dataset) as source:
        count = 0
        for event in source:
            assert event.r0.tels_with_data == {1, 2}
            assert event.count == count
            count += 1

        for event in source:
            # Check generator has restarted from beginning
            assert event.count == 0
            break


def test_that_event_is_not_modified_after_loop():
    dataset = get_dataset_path("20131004_M1_05029747.003_Y_MagicCrab-W0.40+035.root")
    dataset = dataset.replace('_M1_', '_M*_')

    # with MAGICEventSourceROOT(input_url=dataset, max_events=3) as source:
    with MAGICEventSourceROOT(input_url=dataset) as source:
        for event in source:
            last_event = copy.deepcopy(event)

        # now `event` should be identical with the deepcopy of itself from
        # inside the loop.
        # Unfortunately this does not work:
        #      assert last_event == event
        # So for the moment we just compare event ids
        assert event.r0.event_id == last_event.r0.event_id


def test_len():
    dataset = get_dataset_path("20131004_M1_05029747.003_Y_MagicCrab-W0.40+035.root")
    dataset = dataset.replace('_M1_', '_M*_')

    with MAGICEventSourceROOT(input_url=dataset) as source:
        count = 0
        for _ in source:
            count += 1

        # assert count == len(source)
        n_stereo_events = source.current_run['data'].n_stereo_events
        assert count == n_stereo_events

    # with MAGICEventSourceROOT(input_url=dataset, max_events=3) as reader:
    # with MAGICEventSourceROOT(input_url=dataset) as reader:
    #     assert len(reader) == 3


def test_geom():
    dataset = get_dataset_path("20131004_M1_05029747.003_Y_MagicCrab-W0.40+035.root")
    dataset = dataset.replace('_M1_', '_M*_')

    with MAGICEventSourceROOT(input_url=dataset) as source:
        event = next(source._generator())
        assert event.inst.subarray.tels[1].camera.pix_x.size == 1039
        assert event.inst.subarray.tels[2].camera.pix_x.size == 1039


def test_eventsourcefactory():
    dataset = get_dataset_path("20131004_M1_05029747.003_Y_MagicCrab-W0.40+035.root")
    dataset = dataset.replace('_M1_', '_M*_')

    source = EventSourceFactory.produce(input_url=dataset)
    assert source.__class__.__name__ == "MAGICEventSourceROOT"
    assert source.input_url == dataset.replace('_M*_', '_M1_')


def test_eventseeker():
    dataset = get_dataset_path("20131004_M1_05029747.003_Y_MagicCrab-W0.40+035.root")
    dataset = dataset.replace('_M1_', '_M*_')

    with MAGICEventSourceROOT(input_url=dataset) as source:
        seeker = EventSeeker(source)
        event = seeker[0]
        assert event.count == 0
        assert event.dl0.event_id == 29795.0

        event = seeker[2]
        assert event.count == 2
        assert event.r1.event_id == 29798.0

        # event = seeker[-1]
        # assert event.count == len(seeker) - 1

    # with MAGICEventSourceROOT(input_url=dataset, max_events=3) as source:
    #     with pytest.raises(IndexError):
    #         seeker = EventSeeker(source)
    #         seeker[5]
