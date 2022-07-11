import pytest

from ctapipe.instrument.camera import UnknownPixelShapeWarning
from ctapipe.io.simteleventsource import SimTelEventSource
from ctapipe.utils import get_dataset_path

dataset = get_dataset_path("gamma_test.simtel.gz")


def test_eventio_prod2():
    with pytest.warns(UnknownPixelShapeWarning):
        with SimTelEventSource(
            input_url=dataset,
            focal_length_choice="EQUIVALENT",
        ) as reader:
            for event in reader:
                if event.count == 2:
                    break
