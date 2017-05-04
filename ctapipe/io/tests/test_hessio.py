from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils import get_dataset


def test_get_run_id():
    filename = get_dataset("gamma_test.simtel.gz")
    print(filename)
    gen = hessio_event_source(filename)
    event = next(gen)
    tels = event.dl0.tels_with_data
    print(tels)
    assert tels == {38, 47}


def test_get_specific_event():
    dataset = get_dataset("gamma_test.simtel.gz")
    source = hessio_event_source(dataset, requested_event=2)
    event = next(source)
    assert event.count == 2
    assert event.dl0.event_id == 803
    source = hessio_event_source(dataset, requested_event=803,
                                 use_event_id=True)
    event = next(source)
    assert event.count == 2
    assert event.dl0.event_id == 803
