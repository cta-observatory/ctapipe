import pytest
from ctapipe.utils import get_dataset_path
from ctapipe.io.hessioeventsource import HESSIOEventSource
from ctapipe.io.simteleventsource import SimTelEventSource
from ctapipe.instrument.camera import UnknownPixelShapeWarning

dataset = get_dataset_path("gamma_test.simtel.gz")


def test_pyhessio_prod2():
    pytest.importorskip('pyhessio')

    with pytest.warns(UnknownPixelShapeWarning):
        with HESSIOEventSource(input_url=dataset) as reader:
            for event in reader:
                if event.count == 2:
                    break


def test_eventio_prod2():
    with pytest.warns(UnknownPixelShapeWarning):
        with SimTelEventSource(input_url=dataset) as reader:
            for event in reader:
                if event.count == 2:
                    break
