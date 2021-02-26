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
def _subarray_and_event_gamma_off_axis_500_gev():
    from ctapipe.calib import CameraCalibrator
    from ctapipe.image import ImageProcessor

    path = get_dataset_path("lst_prod3_calibration_and_mcphotons.simtel.zst")

    with SimTelEventSource(path) as source:
        it = iter(source)
        # we want the second event, first event is a corner clipper
        next(it)
        event = next(it)

        # make dl1a available
        calib = CameraCalibrator(source.subarray)
        calib(event)

        image_processor = ImageProcessor(
            source.subarray, is_simulation=source.is_simulation
        )

        # make dl1b available
        image_processor(event)
        return source.subarray, event


@pytest.fixture(scope="function")
def subarray_and_event_gamma_off_axis_500_gev(
    _subarray_and_event_gamma_off_axis_500_gev
):
    """
    A four LST subarray event with a nice shower, well suited to test
    reconstruction algorithms.

    This event should be very well reconstructible, as we have four LSTs with
    bright events.

    The event is already calibrated and image parameters have been calculated.

    You can safely mutate the event or subarray in a test as each test
    gets a fresh copy.
    """
    subarray, event = _subarray_and_event_gamma_off_axis_500_gev
    return deepcopy(subarray), deepcopy(event)


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
