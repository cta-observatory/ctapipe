import pytest
from ctapipe.utils import get_dataset_path
from ctapipe.io.eventsource import EventSource


def test_construct():
    with pytest.raises(TypeError):
        EventSource()


class DummyReader(EventSource):
    """
    Simple working EventSource
    """

    def _generator(self):
        return range(len(self.input_url))

    @staticmethod
    def is_compatible(file_path):
        return False


def test_can_be_implemented():
    dataset = get_dataset_path("gamma_test_large.simtel.gz")
    test_reader = DummyReader(input_url=dataset)
    assert test_reader is not None


def test_is_iterable():
    dataset = get_dataset_path("gamma_test_large.simtel.gz")
    test_reader = DummyReader(input_url=dataset)
    for _ in test_reader:
        pass
