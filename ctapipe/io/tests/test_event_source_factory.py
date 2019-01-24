import pytest
from traitlets import TraitError
from ctapipe.io import EventSourceFactory, event_source
from ctapipe.utils import get_dataset_path
from traitlets.config.loader import Config


def test_factory_subclasses():
    factory = EventSourceFactory()
    assert len(factory.__class__.product.values) > 0


def test_factory():
    dataset = get_dataset_path("gamma_test.simtel.gz")
    reader = EventSourceFactory(input_url=dataset).get_product()
    assert reader.__class__.__name__ == "SimTelEventSource"
    assert reader.input_url == dataset


def test_factory_config():
    dataset = get_dataset_path("gamma_test.simtel.gz")
    config = Config()
    config['EventSourceFactory'] = Config()
    config['EventSourceFactory']['input_url'] = dataset
    reader = EventSourceFactory(config=config).get_product()
    assert reader.__class__.__name__ == "SimTelEventSource"
    assert reader.input_url == dataset


def test_factory_different_file():
    dataset = get_dataset_path("gamma_test_large.simtel.gz")
    reader = EventSourceFactory(input_url=dataset).get_product()
    assert reader.__class__.__name__ == "SimTelEventSource"
    assert reader.input_url == dataset


def test_factory_from_reader():
    dataset = get_dataset_path("gamma_test.simtel.gz")
    reader = EventSourceFactory(
        product='SimTelEventSource',
        input_url=dataset
    ).get_product()
    assert reader.__class__.__name__ == "SimTelEventSource"
    assert reader.input_url == dataset


def test_factory_unknown_file_format():
    with pytest.raises(ValueError):
        dataset = get_dataset_path("optics.ecsv.txt")
        reader = EventSourceFactory(input_url=dataset).get_product()
        assert reader is not None


def test_factory_unknown_reader():
    with pytest.raises(TraitError):
        dataset = get_dataset_path("gamma_test.simtel.gz")
        reader = EventSourceFactory(
            product='UnknownFileReader',
            input_url=dataset
        ).get_product()
        assert reader is not None


def test_factory_incompatible_file():
    with pytest.raises(ValueError):
        dataset = get_dataset_path("optics.ecsv.txt")
        EventSourceFactory(input_url=dataset).get_product()


def test_factory_nonexistant_file():
    with pytest.raises(FileNotFoundError):
        dataset = "/fake_path/fake_file.fake_extension"
        reader = EventSourceFactory(input_url=dataset).get_product()
        assert reader is not None


def test_no_file():
    with pytest.raises(ValueError):
        EventSourceFactory().get_product()


def test_factory_incorrect_use():
    # with pytest.raises(FileNotFoundError):
    dataset = get_dataset_path("gamma_test_large.simtel.gz")
    factory = EventSourceFactory(product='SimTelEventSource')
    reader = factory.get_product(input_url=dataset)
    assert reader is not None


def test_event_source_helper():
    path = get_dataset_path("gamma_test_large.simtel.gz")
    with event_source(path) as source:
        for _ in source:
            pass


def test_deprecated_behaviour():
    dataset = get_dataset_path("gamma_test.simtel.gz")
    with pytest.warns(DeprecationWarning):
        reader = EventSourceFactory.produce(input_url=dataset)
    assert reader.__class__.__name__ == "SimTelEventSource"
    assert reader.input_url == dataset

    config = Config()
    config['EventSourceFactory'] = Config()
    config['EventSourceFactory']['input_url'] = dataset
    with pytest.warns(DeprecationWarning):
        reader = EventSourceFactory.produce(config=config)
    assert reader.__class__.__name__ == "SimTelEventSource"
    assert reader.input_url == dataset

