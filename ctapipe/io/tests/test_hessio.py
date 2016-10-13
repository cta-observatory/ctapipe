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
    print(event.meta.mirror_dish_area[38], event.meta.mirror_dish_area[47])
    print(event.meta.mirror_numtiles[38], event.meta.mirror_numtiles[47])
    assert(round(event.meta.mirror_dish_area[38].value, 2) == 14.56)
