"""
common pytest fixtures for tests in ctapipe
"""

import pytest

from copy import deepcopy

from ctapipe.io import SimTelEventSource
from ctapipe.utils import get_dataset_path
from ctapipe.instrument import CameraGeometry


@pytest.fixture(scope="session")
def camera_geometries():
    return [
        CameraGeometry.from_name(name)
        for name in ["LSTCam", "NectarCam", "CHEC", "FlashCam", "MAGICCam"]
    ]


@pytest.fixture(scope="session")
def _global_example_event():
    """
    helper to get a single event from a MC file. Don't use this fixture
    directly, rather use `test_event`
    """
    filename = get_dataset_path("gamma_test_large.simtel.gz")

    print("******************** LOAD TEST EVENT ***********************")

    with SimTelEventSource(input_url=filename) as reader:
        event = next(iter(reader))

    return event


@pytest.fixture(scope="session")
def example_subarray():
    """
    Subarray corresponding to the example event
    """
    filename = get_dataset_path("gamma_test_large.simtel.gz")

    print("******************** LOAD TEST EVENT ***********************")

    with SimTelEventSource(input_url=filename) as reader:
        return reader.subarray


@pytest.fixture(scope="function")
def example_event(_global_example_event):
    """
    Use this fixture anywhere you need a test event read from a MC file. For
    example:

    .. code-block::
        def test_my_thing(test_event):
            assert len(test_event.r0.tel) > 0

    """
    return deepcopy(_global_example_event)


@pytest.fixture(scope="session")
def prod5_gamma_simtel_path():
    return get_dataset_path(
        "gamma_20deg_0deg_run2___cta-prod5-paranal_desert-2147m-Paranal-dark_cone10-100evts.simtel.zst"
    )


@pytest.fixture(scope="session")
def prod5_proton_simtel_path():
    return get_dataset_path(
        "proton_20deg_0deg_run4___cta-prod5-paranal_desert-2147m-Paranal-dark-100evts.simtel.zst"
    )
