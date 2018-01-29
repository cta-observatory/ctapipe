from ctapipe.utils import get_dataset
from ctapipe.io.eventsourcefactory import EventSourceFactory, event_source
import pytest
from traitlets import TraitError


def test_factory_subclasses():
    factory= EventSourceFactory(None,None)
    assert len(factory.subclass_names)>0


def test_factory():
    dataset = get_dataset("gamma_test.simtel.gz")
    reader = EventSourceFactory.produce(None, None, input_url=dataset)
    assert reader.__class__.__name__ == "HESSIOEventSource"
    assert reader.input_url == dataset


def test_factory_different_file():
    dataset = get_dataset("gamma_test_large.simtel.gz")
    reader = EventSourceFactory.produce(None, None, input_url=dataset)
    assert reader.__class__.__name__ == "HESSIOEventSource"
    assert reader.input_url == dataset


def test_factory_from_reader():
    dataset = get_dataset("gamma_test.simtel.gz")
    reader = EventSourceFactory.produce(
        None, None,
        reader='HESSIOEventSource',
        input_url=dataset
    )
    assert reader.__class__.__name__ == "HESSIOEventSource"
    assert reader.input_url == dataset


def test_factory_unknown_file_format():
    with pytest.raises(ValueError):
        dataset = get_dataset("optics.ecsv.txt")
        reader = EventSourceFactory.produce(None, None, input_url=dataset)


def test_factory_unknown_reader():
    with pytest.raises(TraitError):
        dataset = get_dataset("gamma_test.simtel.gz")
        reader = EventSourceFactory.produce(
            None, None,
            reader='UnknownFileReader',
            input_url=dataset
        )


def test_factory_incompatible_file():
    dataset = get_dataset("optics.ecsv.txt")
    reader = EventSourceFactory.produce(
        None, None,
        reader='HESSIOEventSource',
        input_url=dataset
    )
    event_list = [event for event in reader]
    assert len(event_list) == 0
    # TODO: Need better test for this, why does pyhessio not throw an error?


def test_factory_nonexistant_file():
    with pytest.raises(FileNotFoundError):
        dataset = "/fake_path/fake_file.fake_extension"
        reader = EventSourceFactory.produce(
            None, None,
            reader='HESSIOEventSource',
            input_url=dataset
        )


def test_factory_incorrect_use():
    with pytest.raises(FileNotFoundError):
        dataset = get_dataset("gamma_test_large.simtel.gz")
        factory = EventSourceFactory(
            None, None,
            input_url=dataset
        )
        reader = factory.produce(None, None)

def test_event_source_helper():
    with event_source(get_dataset("gamma_test_large.simtel.gz")) as source:
        for event in source:
            pass

