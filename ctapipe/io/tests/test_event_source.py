import pytest
from ctapipe.utils import get_dataset_path
from ctapipe.io.eventsource import EventSource
from ctapipe.io import DataLevel


def test_construct():
    with pytest.raises(TypeError):
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
    def obs_id(self):
        return 1

    @property
    def datalevels(self):
        return (DataLevel.R0, )


def test_can_be_implemented():
    dataset = get_dataset_path("gamma_test_large.simtel.gz")
    test_reader = DummyReader(input_url=dataset)
    assert test_reader is not None


def test_is_iterable():
    dataset = get_dataset_path("gamma_test_large.simtel.gz")
    test_reader = DummyReader(input_url=dataset)
    for _ in test_reader:
        pass
