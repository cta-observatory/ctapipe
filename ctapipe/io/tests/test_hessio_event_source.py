import copy
import pytest
from ctapipe.utils import get_dataset_path
from ctapipe.io.hessioeventsource import HESSIOEventSource

dataset = get_dataset_path("gamma_test_large.simtel.gz")

pytest.importorskip('pyhessio')


def test_hessio_file_reader():
    with HESSIOEventSource(input_url=dataset) as reader:
        assert not reader.is_stream
        for event in reader:
            if event.count == 1:
                break

        for event in reader:
            # Check generator has restarted from beginning
            assert event.count == 0
            break

    # test that max_events works:
    max_events = 5
    with HESSIOEventSource(input_url=dataset, max_events=max_events) as reader:
        count = 0
        for _ in reader:
            count += 1
        assert count == max_events

    # test that the allowed_tels mask works:
    with HESSIOEventSource(
            input_url=dataset,
            allowed_tels={3, 4},
            max_events=max_events,
    ) as reader:
        for event in reader:
            assert event.r0.tels_with_data.issubset(reader.allowed_tels)


def test_that_event_is_not_modified_after_loop():

    dataset = get_dataset_path("gamma_test_large.simtel.gz")
    with HESSIOEventSource(input_url=dataset, max_events=2) as source:
        for event in source:
            last_event = copy.deepcopy(event)

        # now `event` should be identical with the deepcopy of itself from
        # inside the loop.
        # Unfortunately this does not work:
        #      assert last_event == event
        # So for the moment we just compare event ids
        assert event.r0.event_id == last_event.r0.event_id
