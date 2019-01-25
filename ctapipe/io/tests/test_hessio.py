from ctapipe.utils import get_dataset_path
from ctapipe.io.hessioeventsource import HESSIOEventSource


def test_hessio_event_source():
    filename = get_dataset_path("gamma_test.simtel.gz")

    with HESSIOEventSource(input_url=filename) as source:
        event = next(iter(source))
        tels = event.dl0.tels_with_data
        assert tels == {38, 47}
