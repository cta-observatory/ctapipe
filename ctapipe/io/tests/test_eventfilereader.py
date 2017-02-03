from os.path import join
from ctapipe.utils.datasets import get_datasets_path
from ctapipe.io.eventfilereader import EventFileReader, \
    EventFileReaderFactory, HessioFileReader


def test_event_file_reader():
    try:
        EventFileReader(config=None, tool=None)
    except TypeError:
        return
    raise TypeError("EventFileReader should raise a TypeError when "
                    "instantiated due to its abstract methods")


def test_hessio_file_reader():
    dataset = get_datasets_path("gamma_test.simtel.gz")
    file = HessioFileReader(None, None, input_path=dataset)
    datasets_path = get_datasets_path("")
    assert file.input_path == join(datasets_path, "gamma_test.simtel.gz")
    assert file.directory == datasets_path
    assert file.extension == ".gz"
    assert file.filename == "gamma_test.simtel"
    source = file.read()
    event = next(source)
    assert event.dl0.tels_with_data == {38, 47}


def test_get_event():
    dataset = get_datasets_path("gamma_test.simtel.gz")
    file = HessioFileReader(None, None, input_path=dataset)
    event = file.get_event(2)
    assert event.count == 2
    assert event.dl0.event_id == 803
    event = file.get_event(803, True)
    assert event.count == 2
    assert event.dl0.event_id == 803


def test_get_num_events():
    dataset = get_datasets_path("gamma_test.simtel.gz")
    file = HessioFileReader(None, None, input_path=dataset)
    num_events = file.num_events
    assert(num_events == 9)

    file.max_events = 2
    num_events = file.num_events
    assert (num_events == 2)


def test_event_file_reader_factory():
    dataset = get_datasets_path("gamma_test.simtel.gz")
    factory = EventFileReaderFactory(None, None)
    factory.input_path = dataset
    cls = factory.get_class()
    file = cls(None, None)
    num_events = file.num_events
    assert(num_events == 9)
