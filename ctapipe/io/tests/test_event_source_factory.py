import pytest
from ctapipe.io import event_source, eventsource
from ctapipe.utils import get_dataset_path


def test_factory():
    dataset = get_dataset_path("gamma_test.simtel.gz")
    reader = eventsource.for_url(url=dataset)
    assert reader.__class__.__name__ == "SimTelEventSource"
    assert reader.input_url == dataset


def test_factory_different_file():
    dataset = get_dataset_path("gamma_test_large.simtel.gz")
    reader = eventsource.for_url(url=dataset)
    assert reader.__class__.__name__ == "SimTelEventSource"
    assert reader.input_url == dataset


def test_factory_incompatible_file():
    with pytest.raises(ValueError):
        dataset = get_dataset_path("optics.ecsv.txt")
        eventsource.for_url(url=dataset)


def test_factory_nonexistant_file():
    with pytest.raises(FileNotFoundError):
        dataset = "/fake_path/fake_file.fake_extension"
        eventsource.for_url(url=dataset)


def test_event_source_helper():
    with event_source(get_dataset_path("gamma_test_large.simtel.gz")) as source:
        for _ in source:
            pass
