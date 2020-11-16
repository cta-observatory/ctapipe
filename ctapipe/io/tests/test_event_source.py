import pytest
from ctapipe.utils import get_dataset_path
from ctapipe.io import EventSource, SimTelEventSource, DataLevel
from traitlets.config.loader import Config
from traitlets import TraitError

from ctapipe.core import Component


def test_construct():
    # at least one of input_url / parent / config is required
    with pytest.raises(ValueError):
        EventSource()


class DummyReader(EventSource):
    """
    Simple working EventSource
    """

    def _generator(self):
        return range(5)

    @staticmethod
    def is_compatible(file_path):
        return False

    @property
    def subarray(self):
        return None

    @property
    def is_simulation(self):
        return False

    @property
    def obs_ids(self):
        return [1]

    @property
    def datalevels(self):
        return (DataLevel.R0,)


def test_can_be_implemented():
    dataset = get_dataset_path("gamma_test_large.simtel.gz")
    test_reader = DummyReader(input_url=dataset)
    assert test_reader is not None


def test_is_iterable():
    dataset = get_dataset_path("gamma_test_large.simtel.gz")
    test_reader = DummyReader(input_url=dataset)
    for _ in test_reader:
        pass


def test_function():
    dataset = get_dataset_path("gamma_test_large.simtel.gz")
    reader = EventSource(input_url=dataset)
    assert isinstance(reader, SimTelEventSource)
    assert reader.input_url == dataset


def test_function_incompatible_file():
    with pytest.raises(ValueError):
        dataset = get_dataset_path("optics.ecsv.txt")
        EventSource(input_url=dataset)


def test_function_nonexistant_file():
    with pytest.raises(TraitError):
        dataset = "/fake_path/fake_file.fake_extension"
        EventSource(input_url=dataset)


def test_from_config():
    dataset = get_dataset_path("gamma_test_large.simtel.gz")
    config = Config({"EventSource": {"input_url": dataset}})
    reader = EventSource(config=config)
    assert isinstance(reader, SimTelEventSource)
    assert reader.input_url == dataset


def test_from_config_parent():
    dataset = get_dataset_path("gamma_test_large.simtel.gz")

    class Parent(Component):
        def __init__(self, config=None, parent=None):
            super().__init__(config=config, parent=parent)

            self.source = EventSource.from_config(parent=self)

    # test with EventSource in root of config
    config = Config({"EventSource": {"input_url": dataset}})

    parent = Parent(config=config)

    assert isinstance(parent.source, SimTelEventSource)
    assert parent.source.parent is parent

    # test with EventSource as subconfig of parent
    config = Config({"Parent": {"EventSource": {"input_url": dataset}}})

    parent = Parent(config=config)
    assert isinstance(parent.source, SimTelEventSource)
    assert parent.source.parent is parent


def test_from_config_default():
    old_default = EventSource.input_url.default_value
    dataset = get_dataset_path("gamma_test_large.simtel.gz")
    EventSource.input_url.default_value = dataset
    config = Config()
    reader = EventSource(config=config, parent=None)
    assert isinstance(reader, SimTelEventSource)
    assert reader.input_url == dataset
    EventSource.input_url.default_value = old_default


def test_from_config_invalid_type():
    dataset = get_dataset_path("gamma_test_large.simtel.gz")
    EventSource.input_url.default_value = dataset
    config = Config({"EventSource": {"input_url": 124}})
    with pytest.raises(TraitError):
        EventSource(config=config, parent=None)


def test_event_source_input_url_config_override():
    dataset1 = get_dataset_path("gamma_test_large.simtel.gz")
    dataset2 = get_dataset_path(
        "gamma_LaPalma_baseline_20Zd_180Az_prod3b_test.simtel.gz"
    )

    config = Config({"EventSource": {"input_url": dataset1}})
    reader = EventSource(input_url=dataset2, config=config)

    assert isinstance(reader, SimTelEventSource)
    assert reader.input_url == dataset2


def test_max_events():
    max_events = 10
    dataset = get_dataset_path("gamma_test_large.simtel.gz")
    reader = EventSource(input_url=dataset, max_events=max_events)
    assert reader.max_events == max_events


def test_max_events_from_config():
    dataset = get_dataset_path("gamma_test_large.simtel.gz")
    max_events = 10
    config = Config({"EventSource": {"input_url": dataset, "max_events": max_events}})
    reader = EventSource(config=config)
    assert reader.max_events == max_events


def test_allowed_tels():
    dataset = get_dataset_path("gamma_test_large.simtel.gz")
    reader = EventSource(input_url=dataset)
    assert reader.allowed_tels is None
    reader = EventSource(input_url=dataset, allowed_tels={1, 3})
    assert len(reader.allowed_tels) == 2


def test_allowed_tels_from_config():
    dataset = get_dataset_path("gamma_test_large.simtel.gz")
    config = Config({"EventSource": {"input_url": dataset, "allowed_tels": {1, 3}}})
    reader = EventSource(config=config, parent=None)
    assert len(reader.allowed_tels) == 2
