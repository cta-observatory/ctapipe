from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils.datasets import get_datasets_path


def test_get_run_id():
    filename = get_datasets_path("gamma_test.simtel.gz")
    print(filename)
    gen = hessio_event_source(filename)
    event = next(gen)
    tels = event.dl0.tels_with_data
    print(tels)
    assert tels == {38, 47}
