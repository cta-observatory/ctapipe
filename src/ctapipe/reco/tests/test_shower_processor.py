"""
Tests for ShowerProcessor functionalities.
"""
from copy import deepcopy

import pytest
from numpy import isfinite
from traitlets.config.loader import Config

from ctapipe.calib import CameraCalibrator
from ctapipe.image import ImageProcessor
from ctapipe.reco import ShowerProcessor
from ctapipe.utils import get_dataset_path

SIMTEL_PATH = get_dataset_path(
    "gamma_20deg_0deg_run2___cta-prod5-paranal_desert"
    "-2147m-Paranal-dark_cone10-100evts.simtel.zst"
)


def get_simtel_profile_from_eventsource():
    """get a TableAtmosphereDensityProfile from a simtel file"""
    from ctapipe.io import EventSource

    with EventSource(SIMTEL_PATH) as source:
        return source.atmosphere_density_profile


@pytest.fixture(scope="session")
def table_profile():
    """a table profile for testing"""
    return get_simtel_profile_from_eventsource()


@pytest.mark.parametrize(
    "reconstructor_types",
    [
        ["HillasIntersection"],
        ["HillasReconstructor"],
        pytest.param("ImPACTReconstructor", marks=pytest.mark.xfail),
    ],
    ids=["HillasIntersection", "HillasReconstructor", "ImPACTReconstructor"],
)
def test_shower_processor_geometry(
    example_event, example_subarray, reconstructor_types, table_profile
):
    """Ensure we get shower geometry when we input an event with parametrized images."""

    calibrate = CameraCalibrator(subarray=example_subarray)

    config = Config()

    process_images = ImageProcessor(
        subarray=example_subarray, image_cleaner_type="MARSImageCleaner"
    )

    process_shower = ShowerProcessor(
        subarray=example_subarray,
        atmosphere_profile=table_profile,
        reconstructor_types=reconstructor_types,
    )

    calibrate(example_event)
    process_images(example_event)

    example_event_copy = deepcopy(example_event)

    process_shower(example_event_copy)

    print(reconstructor_types)
    for reco_type in reconstructor_types:
        DL2a = example_event_copy.dl2.stereo.geometry[reco_type]
        print(reco_type)
        print(DL2a)
        assert isfinite(DL2a.alt)
        assert isfinite(DL2a.az)
        assert isfinite(DL2a.core_x)
        assert isfinite(DL2a.core_x)
        assert isfinite(DL2a.core_y)
        assert DL2a.is_valid
        assert isfinite(DL2a.average_intensity)

    # Increase some quality cuts and check that we get defaults
    for reco_type in reconstructor_types:
        config[reco_type].StereoQualityQuery.quality_criteria = [
            ("> 500 phes", "parameters.hillas.intensity > 500")
        ]

    process_shower = ShowerProcessor(
        config=config,
        subarray=example_subarray,
        atmosphere_profile=table_profile,
        reconstructor_types=reconstructor_types,
    )

    example_event_copy = deepcopy(example_event)
    process_shower(example_event_copy)

    for reco_type in reconstructor_types:
        DL2a = example_event_copy.dl2.stereo.geometry[reco_type]
        print(DL2a)
        assert not isfinite(DL2a.alt)
        assert not isfinite(DL2a.az)
        assert not isfinite(DL2a.core_x)
        assert not isfinite(DL2a.core_x)
        assert not isfinite(DL2a.core_y)
        assert not DL2a.is_valid
        assert not isfinite(DL2a.average_intensity)
