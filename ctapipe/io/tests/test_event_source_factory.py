import pytest
from traitlets import TraitError

from ctapipe.io import EventSourceFactory, event_source
from ctapipe.utils import get_dataset_path


def test_factory_subclasses():
    factory = EventSourceFactory()
    assert len(factory.__class__.product.values) > 0


def test_factory():
    dataset = get_dataset_path("gamma_test.simtel.gz")
    reader = EventSourceFactory(input_url=dataset).produce()
    assert reader.__class__.__name__ == "SimTelEventSource"
    assert reader.input_url == dataset


def test_factory_different_file():
    dataset = get_dataset_path("gamma_test_large.simtel.gz")
    reader = EventSourceFactory(input_url=dataset).produce()
    assert reader.__class__.__name__ == "SimTelEventSource"
    assert reader.input_url == dataset


def test_factory_from_reader():
    dataset = get_dataset_path("gamma_test.simtel.gz")
    reader = EventSourceFactory(
        product='SimTelEventSource',
        input_url=dataset
    ).produce()
    assert reader.__class__.__name__ == "SimTelEventSource"
    assert reader.input_url == dataset


def test_factory_unknown_file_format():
    with pytest.raises(ValueError):
        dataset = get_dataset_path("optics.ecsv.txt")
        reader = EventSourceFactory(input_url=dataset).produce()
        assert reader is not None


def test_factory_unknown_reader():
    with pytest.raises(TraitError):
        dataset = get_dataset_path("gamma_test.simtel.gz")
        reader = EventSourceFactory(
            product='UnknownFileReader',
            input_url=dataset
        ).produce()
        assert reader is not None


def test_factory_incompatible_file():
    with pytest.raises(ValueError):
        dataset = get_dataset_path("optics.ecsv.txt")
        EventSourceFactory(input_url=dataset).produce()


def test_factory_nonexistant_file():
    with pytest.raises(FileNotFoundError):
        dataset = "/fake_path/fake_file.fake_extension"
        reader = EventSourceFactory(input_url=dataset).produce()
        assert reader is not None


def test_no_file():
    with pytest.raises(ValueError):
        EventSourceFactory().produce()


def test_factory_incorrect_use():
    # with pytest.raises(FileNotFoundError):
    dataset = get_dataset_path("gamma_test_large.simtel.gz")
    factory = EventSourceFactory(product='SimTelEventSource')
    reader = factory.produce(input_url=dataset)
    assert reader is not None


def test_event_source_helper():
    path = get_dataset_path("gamma_test_large.simtel.gz")
    with event_source(path) as source:
        for _ in source:
            pass
