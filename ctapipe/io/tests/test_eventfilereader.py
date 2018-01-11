from os.path import join, dirname
from ctapipe.utils import get_dataset
from ctapipe.io.eventfilereader import EventFileReader, \
    EventFileReaderFactory


def test_event_file_reader():
    try:
        EventFileReader(config=None, tool=None)
    except TypeError:
        return
    raise TypeError("EventFileReader should raise a TypeError when "
                    "instantiated due to its abstract methods")


def test_event_file_reader_factory():
    dataset = get_dataset("gamma_test.simtel.gz")
    factory = EventFileReaderFactory(None, None)
    factory.input_path = dataset
    cls = factory.get_class()
    reader = cls(None, None)
    assert reader.__class__.__name__ == "HessioFileReader"
