from ctapipe.io.eventsource import EventSource
from ctapipe.utils import get_dataset_path


def test_construct():
    try:
        EventSource(config=None, tool=None)
    except TypeError:
        return
    raise TypeError("EventSource should raise a TypeError when "
                    "instantiated due to its abstract methods")


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
    dataset = get_dataset_path("gamma_test.simtel.gz")
    test_reader = DummyReader(input_url=dataset)
    assert test_reader is not None


def test_is_iterable():
    dataset = get_dataset_path("gamma_test.simtel.gz")
    test_reader = DummyReader(input_url=dataset)
    for _ in test_reader:
        pass
