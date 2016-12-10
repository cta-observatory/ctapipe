from os.path import join
from ctapipe.utils.datasets import get_datasets_path
from ctapipe.io.files import get_file_type, FileReader


def test_get_file_type():

    test_filenames = {'test.fits.gz': 'fits',
                      'test.fits': 'fits',
                      'test.fits.bz2': 'fits',
                      'test.fit': 'fits',
                      'test_file.eventio.gz': 'eventio',
                      'test_file.eventio': 'eventio',
                      'more.complex.fileame.txt.gz': 'txt'}

    for filename, filetype in test_filenames.items():
        assert get_file_type(filename) == filetype


def test_filereader():
    dataset = get_datasets_path("gamma_test.simtel.gz")
    file = FileReader(None, None, input_path=dataset, source='hessio')
    datasets_path = get_datasets_path("")
    assert file.input_path == join(datasets_path, "gamma_test.simtel.gz")
    assert file.directory == datasets_path
    assert file.extension == ".gz"
    assert file.filename == "gamma_test.simtel"
    assert file.source == "hessio"
    source = file.read()
    event = next(source)
    assert event.dl0.tels_with_data == {38, 47}


def test_getevent():
    dataset = get_datasets_path("gamma_test.simtel.gz")
    file = FileReader(None, None, input_path=dataset, source='hessio')
    event = file.get_event(2)
    assert event.count == 2
    assert event.dl0.event_id == 803
    event = file.get_event(803, True)
    assert event.count == 2
    assert event.dl0.event_id == 803


def test_get_num_events():
    dataset = get_datasets_path("gamma_test.simtel.gz")
    file = FileReader(None, None, input_path=dataset, source='hessio')
    num_events = file.num_events
    assert(num_events == 9)

    file.max_events = 2
    num_events = file.num_events
    assert (num_events == 2)
