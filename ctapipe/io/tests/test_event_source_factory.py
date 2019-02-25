import pytest
from traitlets.config.loader import Config
from traitlets import TraitError
from ctapipe.io import event_source, SimTelEventSource
from ctapipe.io.eventsource import EventSource
from ctapipe.utils import get_dataset_path


def test_factory():
    dataset = get_dataset_path("gamma_test_large.simtel.gz")
    reader = event_source(input_url=dataset)
    assert isinstance(reader, SimTelEventSource)
    assert reader.input_url == dataset


def test_factory_different_file():
    dataset = get_dataset_path("gamma_test_large.simtel.gz")
    reader = event_source(input_url=dataset)
    assert isinstance(reader, SimTelEventSource)
    assert reader.input_url == dataset


def test_factory_incompatible_file():
    with pytest.raises(ValueError):
        dataset = get_dataset_path("optics.ecsv.txt")
        event_source(input_url=dataset)


def test_factory_nonexistant_file():
    with pytest.raises(FileNotFoundError):
        dataset = "/fake_path/fake_file.fake_extension"
        event_source(input_url=dataset)


def test_from_config():
    dataset = get_dataset_path("gamma_test_large.simtel.gz")
    config = Config({'EventSource': {'input_url': dataset}})
    reader = EventSource.from_config(config=config, tool=None)
    assert isinstance(reader, SimTelEventSource)
    assert reader.input_url == dataset


def test_from_config_default():
    old_default = EventSource.input_url.default_value
    dataset = get_dataset_path("gamma_test_large.simtel.gz")
    EventSource.input_url.default_value = dataset
    config = Config()
    reader = EventSource.from_config(config=config, tool=None)
    assert isinstance(reader, SimTelEventSource)
    assert reader.input_url == dataset
    EventSource.input_url.default_value = old_default


def test_from_config_invalid_type():
    dataset = get_dataset_path("gamma_test_large.simtel.gz")
    EventSource.input_url.default_value = dataset
    config = Config({'EventSource': {'input_url': 124}})
    with pytest.raises(TraitError):
        EventSource.from_config(config=config, tool=None)


def test_event_source_config():
    dataset1 = get_dataset_path("gamma_test_large.simtel.gz")
    dataset2 = get_dataset_path("gamma_test_large.simtel.gz")
    config = Config({'EventSource': {'input_url': dataset1}})
    reader = event_source(dataset2, config=config)
    assert isinstance(reader, SimTelEventSource)
    assert reader.input_url == dataset2


def test_event_source_input_url_config_override():
    dataset1 = get_dataset_path("gamma_test_large.simtel.gz")
    dataset2 = get_dataset_path("gamma_test_large.simtel.gz")
    config = Config({'EventSource': {'input_url': dataset1}})
    reader = event_source(input_url=dataset2, config=config)
    assert isinstance(reader, SimTelEventSource)
    assert reader.input_url == dataset2


def test_factory_max_events():
    max_events = 10
    dataset = get_dataset_path("gamma_test_large.simtel.gz")
    reader = event_source(input_url=dataset, max_events=max_events)
    assert reader.max_events == max_events


def test_factory_max_events_from_config():
    dataset = get_dataset_path("gamma_test_large.simtel.gz")
    max_events = 10
    config = Config({'EventSource': {
        'input_url': dataset,
        'max_events': max_events,
    }})
    reader = EventSource.from_config(config=config)
    assert reader.max_events == max_events


def test_factory_allowed_tels():
    dataset = get_dataset_path("gamma_test_large.simtel.gz")
    reader = event_source(input_url=dataset)
    assert len(reader.allowed_tels) == 0
    reader = event_source(input_url=dataset, allowed_tels={1, 3})
    assert len(reader.allowed_tels) == 2


def test_factory_allowed_tels_from_config():
    dataset = get_dataset_path("gamma_test_large.simtel.gz")
    config = Config({'EventSource': {
        'input_url': dataset,
        'allowed_tels': {1, 3}
    }})
    reader = EventSource.from_config(config=config, tool=None)
    assert len(reader.allowed_tels) == 2
