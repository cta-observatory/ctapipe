from ctapipe.utils import get_dataset_path
from ctapipe.io import SimTelEventSource
from ctapipe.io.eventseeker import EventSeeker
import pytest

dataset = get_dataset_path("gamma_test_large.simtel.gz")


def test_eventseeker():

    with SimTelEventSource(input_url=dataset, back_seekable=True) as reader:

        seeker = EventSeeker(reader=reader)

        event = seeker[1]
        assert event.count == 1
        event = seeker[0]
        assert event.count == 0

        event = seeker['31007']
        assert event.r0.event_id == 31007

        events = seeker[0:2]

        for i, event in enumerate(events):
            assert event.count == i

        events = seeker[[0, 1]]
        for i, event in enumerate(events):
            assert event.count == i

        ids = ['23703', '31007']
        events = seeker[ids]

        for i, event in zip(ids, events):
            assert event.r0.event_id == int(i)

        with pytest.raises(IndexError):
            event = seeker[200]

        with pytest.raises(ValueError):
            event = seeker['t']

        with pytest.raises(TypeError):
            event = seeker[dict()]

    with SimTelEventSource(input_url=dataset, max_events=5, back_seekable=True) as reader:
        seeker = EventSeeker(reader=reader)
        with pytest.raises(IndexError):
            event = seeker[5]
            assert event is not None

    class StreamFileReader(SimTelEventSource):
        def is_stream(self):
            return True

    with StreamFileReader(input_url=dataset) as reader:
        with pytest.raises(IOError):
            seeker = EventSeeker(reader=reader)
