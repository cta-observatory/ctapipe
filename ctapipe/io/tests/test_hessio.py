from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils import get_dataset_path
from ctapipe.io.hessioeventsource import HESSIOEventSource


def test_hessio_event_source():
    filename = get_dataset_path("gamma_test.simtel.gz")
    source = hessio_event_source(filename)
    event = next(source)
    tels = event.dl0.tels_with_data
    print(tels)
    assert tels == {38, 47}

