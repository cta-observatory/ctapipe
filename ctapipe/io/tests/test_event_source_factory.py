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


def test_factory_max_events():
    max_events = 10
    dataset = get_dataset_path("gamma_test.simtel.gz")
    reader = EventSourceFactory(
        input_url=dataset, max_events=max_events
    ).get_product()
    assert reader.max_events == max_events


def test_factory_max_events_config():
    max_events = 10
    config = Config()
    config['EventSource'] = Config()
    config['EventSource']['max_events'] = max_events
    dataset = get_dataset_path("gamma_test.simtel.gz")
    reader = EventSourceFactory(
        input_url=dataset, config=config,
    ).get_product()
    assert reader.max_events == max_events


def test_factory_allowed_tels():
    dataset = get_dataset_path("gamma_test.simtel.gz")
    reader = EventSourceFactory(
        input_url=dataset,
    ).get_product()
    assert len(reader.allowed_tels) == 0
    reader = EventSourceFactory(
        input_url=dataset, allowed_tels={1, 3}
    ).get_product()
    assert len(reader.allowed_tels) == 2


def test_factory_allowed_tels_config():
    config = Config()
    config['EventSource'] = Config()
    config['EventSource']['allowed_tels'] = {1, 3}
    dataset = get_dataset_path("gamma_test.simtel.gz")
    reader = EventSourceFactory(
        input_url=dataset, config=config,
    ).get_product()
    assert len(reader.allowed_tels) == 2


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


def test_factory_unknown_file_format():
    with pytest.raises(ValueError):
        dataset = get_dataset_path("optics.ecsv.txt")
        reader = EventSourceFactory(input_url=dataset).get_product()
        assert reader is not None


def test_factory_from_product():
    dataset = get_dataset_path("gamma_test.simtel.gz")
    reader = EventSourceFactory(
        input_url=dataset,
        product="HESSIOEventSource",
    ).get_product()
    assert reader.__class__.__name__ == "HESSIOEventSource"


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


def test_event_source_helper():
    path = get_dataset_path("gamma_test_large.simtel.gz")
    with event_source(path) as source:
        assert source.__class__.__name__ == "SimTelEventSource"
        assert source.input_url == path


def test_event_source_helper_max_events():
    max_events = 10
    path = get_dataset_path("gamma_test_large.simtel.gz")
    with event_source(path, max_events=max_events) as source:
        assert source.max_events == max_events


def test_event_source_helper_allowed_tels():
    path = get_dataset_path("gamma_test_large.simtel.gz")
    with event_source(path) as source:
        assert len(source.allowed_tels) == 0
    with event_source(path, allowed_tels={1, 3}) as source:
        assert len(source.allowed_tels) == 2


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

    dataset = get_dataset_path("gamma_test.simtel.gz")
    with pytest.warns(DeprecationWarning):
        with pytest.raises(SyntaxError):
            EventSourceFactory(input_url=dataset).produce()
