from os.path import join, dirname
from ctapipe.utils import get_dataset
from ctapipe.io.eventfilereader import EventFileReader, \
    EventFileReaderFactory
import pytest
from traitlets import TraitError


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
    reader = EventFileReaderFactory.produce(None, None, input_url=dataset)
    assert reader.__class__.__name__ == "HessioFileReader"
    assert reader.input_url == dataset


def test_event_file_reader_factory_different_file():
    dataset = get_dataset("gamma_test_large.simtel.gz")
    reader = EventFileReaderFactory.produce(None, None, input_url=dataset)
    assert reader.__class__.__name__ == "HessioFileReader"
    assert reader.input_url == dataset


def test_event_file_reader_factory_from_reader():
    dataset = get_dataset("gamma_test.simtel.gz")
    reader = EventFileReaderFactory.produce(
        None, None,
        reader='HessioFileReader',
        input_url=dataset
    )
    assert reader.__class__.__name__ == "HessioFileReader"
    assert reader.input_url == dataset


def test_event_file_reader_factory_unknown_file_format():
    with pytest.raises(ValueError):
        dataset = get_dataset("optics.ecsv.txt")
        reader = EventFileReaderFactory.produce(None, None, input_url=dataset)


def test_event_file_reader_factory_unknown_reader():
    with pytest.raises(TraitError):
        dataset = get_dataset("gamma_test.simtel.gz")
        reader = EventFileReaderFactory.produce(
            None, None,
            reader='UnknownFileReader',
            input_url=dataset
        )


def test_event_file_reader_factory_incompatible_file():
    dataset = get_dataset("optics.ecsv.txt")
    reader = EventFileReaderFactory.produce(
        None, None,
        reader='HessioFileReader',
        input_url=dataset
    )
    event_list = [event for event in reader]
    assert len(event_list) == 0
    # TODO: Need better test for this, why does pyhessio not throw an error?


def test_event_file_reader_factory_nonexistant_file():
    with pytest.raises(FileNotFoundError):
        dataset = "/fake_path/fake_file.fake_extension"
        reader = EventFileReaderFactory.produce(
            None, None,
            reader='HessioFileReader',
            input_url=dataset
        )


def test_event_file_reader_factory_incorrect_use():
    with pytest.raises(AssertionError):
        dataset = get_dataset("gamma_test_large.simtel.gz")
        factory = EventFileReaderFactory(
            None, None,
            input_url=dataset
        )
        reader = factory.produce(None, None)
        assert reader.input_url == dataset
