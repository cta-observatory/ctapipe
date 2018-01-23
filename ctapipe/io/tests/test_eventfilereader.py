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


class TestReader(EventFileReader):
    """
    Simple working EventFileReader
    """
    def _generator(self):
        return range(len(self.input_url))

    def is_compatible(self, path):
        return False


def test_can_be_implemented():
    dataset = get_dataset("gamma_test.simtel.gz")
    test_reader = TestReader(None, None, input_url=dataset)


def test_is_iterable():
    dataset = get_dataset("gamma_test.simtel.gz")
    test_reader = TestReader(None, None, input_url=dataset)
    for _ in test_reader:
        pass


def test_event_file_reader_factory():
    dataset = get_dataset("gamma_test.simtel.gz")
    factory = EventFileReaderFactory(None, None)
    factory.input_url = dataset
    cls = factory.get_class()
    reader = cls(None, None)
    assert reader.__class__.__name__ == "HessioFileReader"
