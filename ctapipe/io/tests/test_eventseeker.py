import pytest

from ctapipe.io import EventSource, SimTelEventSource
from ctapipe.io.eventseeker import EventSeeker
from ctapipe.utils import get_dataset_path

dataset = get_dataset_path("gamma_test_large.simtel.gz")


def test_eventseeker(dl1_file):

    with EventSource(dl1_file) as source:
        seeker = EventSeeker(event_source=source)

        event = seeker.get_event_index(1)
        assert event.count == 1
        event = seeker.get_event_index(0)
        assert event.count == 0

        event = seeker.get_event_id(5103)
        assert event.index.event_id == 5103

        with pytest.raises(IndexError):
            seeker.get_event_index(200)

        with pytest.raises(TypeError):
            seeker.get_event_index("1")

        with pytest.raises(TypeError):
            seeker.get_event_index("t")

        with pytest.raises(TypeError):
            seeker.get_event_index(dict())

    with EventSource(input_url=dl1_file, max_events=5) as source:
        seeker = EventSeeker(event_source=source)
        with pytest.raises(IndexError):
            event = seeker.get_event_index(5)
            assert event is not None


def test_eventseeker_edit(dl1_file):
    with EventSource(input_url=dl1_file) as reader:
        seeker = EventSeeker(event_source=reader)
        event = seeker.get_event_index(1)
        assert event.count == 1
        event.count = 2
        assert event.count == 2
        event = seeker.get_event_index(1)
        assert event.count == 1


def test_eventseeker_simtel():
    # Ensure the EventSeeker can forward seek even if back-seeking is not possible
    with SimTelEventSource(
        input_url=dataset,
        focal_length_choice="EQUIVALENT",
    ) as reader:
        seeker = EventSeeker(event_source=reader)
        event = seeker.get_event_index(1)
        assert event.count == 1
        event = seeker.get_event_index(1)
        assert event.count == 1
        event = seeker.get_event_index(2)
        assert event.count == 2
        event = seeker.get_event_index(2)
        assert event.count == 2
        event = seeker.get_event_index(4)
        assert event.count == 4
        with pytest.raises(IOError):
            seeker.get_event_index(1)
        event = seeker.get_event_index(5)
        assert event.count == 5
