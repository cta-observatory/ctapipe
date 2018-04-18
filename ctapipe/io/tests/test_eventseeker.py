from ctapipe.utils import get_dataset_path
from ctapipe.io.hessioeventsource import HESSIOEventSource
from ctapipe.io.eventseeker import EventSeeker
import pytest


def test_eventseeker():
    dataset = get_dataset_path("gamma_test.simtel.gz")
    kwargs = dict(config=None, tool=None, input_url=dataset)
    with HESSIOEventSource(**kwargs) as reader:
        seeker = EventSeeker(reader=reader)
        event = seeker[1]
        assert event.r0.tels_with_data == {11, 21, 24, 26, 61, 63, 118, 119}
        event = seeker[0]
        assert event.r0.tels_with_data == {38, 47}
        event = seeker['409']
        assert event.r0.tels_with_data == {11, 21, 24, 26, 61, 63, 118, 119}
        tel_list = [{38, 47}, {11, 21, 24, 26, 61, 63, 118, 119}]
        events = seeker[0:2]
        events_tels = [e.r0.tels_with_data for e in events]
        assert events_tels == tel_list
        events = seeker[[0, 1]]
        events_tels = [e.r0.tels_with_data for e in events]
        assert events_tels == tel_list
        events = seeker[['408', '409']]
        events_tels = [e.r0.tels_with_data for e in events]
        assert events_tels == tel_list
        assert len(seeker) == 9

        with pytest.raises(IndexError):
            event = seeker[200]
            assert event is not None
        with pytest.raises(ValueError):
            event = seeker['t']
            assert event is not None
        with pytest.raises(TypeError):
            event = seeker[dict()]
            assert event is not None

    with HESSIOEventSource(**kwargs, max_events=5) as reader:
        seeker = EventSeeker(reader=reader)
        with pytest.raises(IndexError):
            event = seeker[5]
            assert event is not None

    class StreamFileReader(HESSIOEventSource):

        def is_stream(self):
            return True
    with StreamFileReader(**kwargs) as reader:
        with pytest.raises(IOError):
            seeker = EventSeeker(reader=reader)
            assert seeker is not None
