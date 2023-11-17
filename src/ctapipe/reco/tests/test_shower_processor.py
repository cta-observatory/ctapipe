"""
Tests for ShowerProcessor functionalities.
"""
from copy import deepcopy

import pytest
from numpy import isfinite
from traitlets.config.loader import Config

from ctapipe.calib import CameraCalibrator
from ctapipe.image import ImageProcessor
from ctapipe.reco import HillasGeometryReconstructor, ShowerProcessor


@pytest.mark.parametrize(
    "reconstructor_types",
    [
        [reco_type]
        for reco_type in HillasGeometryReconstructor.non_abstract_subclasses().keys()
    ]
    + [["HillasReconstructor", "HillasIntersection"]],
)
def test_shower_processor_geometry(
    example_event, example_subarray, reconstructor_types
):
    """Ensure we get shower geometry when we input an event with parametrized images."""

    calibrate = CameraCalibrator(subarray=example_subarray)

    config = Config()

    process_images = ImageProcessor(
        subarray=example_subarray, image_cleaner_type="MARSImageCleaner"
    )

    process_shower = ShowerProcessor(
        subarray=example_subarray, reconstructor_types=reconstructor_types
    )

    calibrate(example_event)
    process_images(example_event)

    example_event_copy = deepcopy(example_event)

    process_shower(example_event_copy)

    for reco_type in reconstructor_types:
        DL2a = example_event_copy.dl2.stereo.geometry[reco_type]
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
