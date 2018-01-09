from os.path import join, dirname
from ctapipe.utils import get_dataset
from ctapipe.io.hessiofilereader import HessioFileReader


def test_hessio_file_reader():
    dataset = get_dataset("gamma_test.simtel.gz")
    reader = HessioFileReader(None, None, input_path=dataset)
    assert reader.is_compatible(dataset)
    assert reader.camera == 'hessio'
    assert len(reader) == 9
    for event in reader:
        if event.count == 0:
            assert event.r0.tels_with_data == {38, 47}
        if event.count == 1:
            assert event.r0.tels_with_data == {11, 21, 24, 26, 61, 63, 118,
                                               119}
    event = reader[0]
    assert event.r0.tels_with_data == {38, 47}
    event = reader['409']
    assert event.r0.tels_with_data == {11, 21, 24, 26, 61, 63, 118, 119}
    reader = HessioFileReader(None, None, input_path=dataset, max_events=5)
    assert len(reader) == 5
