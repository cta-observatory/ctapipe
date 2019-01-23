import pytest
from ctapipe.io import event_source
from ctapipe.utils import get_dataset_path


def test_factory():
    dataset = get_dataset_path("gamma_test.simtel.gz")
    reader = event_source(url=dataset)
    assert reader.__class__.__name__ == "SimTelEventSource"
    assert reader.input_url == dataset


def test_factory_different_file():
    dataset = get_dataset_path("gamma_test_large.simtel.gz")
    reader = event_source(url=dataset)
    assert reader.__class__.__name__ == "SimTelEventSource"
    assert reader.input_url == dataset


def test_factory_incompatible_file():
    with pytest.raises(ValueError):
        dataset = get_dataset_path("optics.ecsv.txt")
        event_source(url=dataset)


def test_factory_nonexistant_file():
    with pytest.raises(FileNotFoundError):
        dataset = "/fake_path/fake_file.fake_extension"
        event_source(url=dataset)
