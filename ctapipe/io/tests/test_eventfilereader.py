import pytest
from os.path import dirname
from ctapipe.utils import get_dataset
from ctapipe.io.eventfilereader import EventFileReader, \
    EventFileReaderFactory, HessioFileReader
from ctapipe.io.unofficial import eventfilereader as uefr


def test_event_file_reader():
    try:
        EventFileReader(config=None, tool=None)
    except TypeError:
        return
    raise TypeError("EventFileReader should raise a TypeError when "
                    "instantiated due to its abstract methods")


def test_hessio_file_reader():
    dataset = get_dataset("gamma_test.simtel.gz")
    file = HessioFileReader(None, None, input_path=dataset)
    assert file.directory == dirname(dataset)
    assert file.extension == ".gz"
    assert file.filename == "gamma_test.simtel"
    source = file.read()
    event = next(source)
    assert event.r0.tels_with_data == {38, 47}


def test_get_event():
    dataset = get_dataset("gamma_test.simtel.gz")
    file = HessioFileReader(None, None, input_path=dataset)
    event = file.get_event(2)
    assert event.count == 2
    assert event.r0.event_id == 803
    event = file.get_event(803, True)
    assert event.count == 2
    assert event.r0.event_id == 803


def test_get_num_events():
    dataset = get_dataset("gamma_test.simtel.gz")
    file = HessioFileReader(None, None, input_path=dataset)
    num_events = file.num_events
    assert(num_events == 9)

    file.max_events = 2
    num_events = file.num_events
    assert (num_events == 2)


def test_event_file_reader_factory_hessio():
    dataset = get_dataset("gamma_test.simtel.gz")
    factory = EventFileReaderFactory(None, None)
    factory.input_path = dataset
    cls = factory.get_class()
    file = cls(None, None)
    assert(file.origin == 'hessio')
    assert(file.num_events == 9)


@pytest.mark.skipif(not uefr.check_modules_installed(uefr.targetio_modules),
                    reason="Requires targetio specific modules")
def test_event_file_reader_factory_targetio():
    dataset = get_dataset("chec_r1.tio")
    factory = EventFileReaderFactory(None, None)
    factory.input_path = dataset
    cls = factory.get_class()
    file = cls(None, None)
    assert(file.origin == 'targetio')
    assert(file.num_events == 5)
