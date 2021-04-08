"""
Tests for ShowerProcessor functionalities.
"""
import pytest
from numpy import isfinite

from traitlets.config.loader import Config

from ctapipe.calib import CameraCalibrator
from ctapipe.image import ImageProcessor
from ctapipe.reco import ShowerProcessor


def test_shower_processor_geometry(example_event, example_subarray):
    """Ensure we get shower geometry when we input an event with parametrized images."""

    calibrate = CameraCalibrator(subarray=example_subarray)

    config = Config()
    config.ShowerProcessor.reconstruct_energy = False
    config.ShowerProcessor.classify = False

    process_images = ImageProcessor(
        subarray=example_subarray,
        is_simulation=True,
        image_cleaner_type="MARSImageCleaner",
    )

    process_shower = ShowerProcessor(
        subarray=example_subarray,
        reconstruct_energy=False,
        classify=False
    )

    calibrate(example_event)
    process_images(example_event)

    process_shower(example_event)
    print(process_shower.check_shower.to_table())

    DL2a = example_event.dl2.shower["HillasReconstructor"]
    print(DL2a)
    assert isfinite(DL2a.alt)
    assert isfinite(DL2a.az)
    assert isfinite(DL2a.core_x)
    assert isfinite(DL2a.core_x)
    assert isfinite(DL2a.core_y)
    assert DL2a.is_valid
    assert isfinite(DL2a.average_intensity)

    # Increase some quality cuts and check that we get defaults
    config.ShowerQualityQuery.quality_criteria = [("> 500 phes", "lambda p: p.hillas.intensity > 500")]

    process_shower = ShowerProcessor(
        config=config,
        subarray=example_subarray,
        reconstruct_energy=False,
        classify=False
    )

    process_shower(example_event)
    print(process_shower.check_shower.to_table())

    DL2a = example_event.dl2.shower["HillasReconstructor"]
    print(DL2a)
    assert not isfinite(DL2a.alt)
    assert not isfinite(DL2a.az)
    assert not isfinite(DL2a.core_x)
    assert not isfinite(DL2a.core_x)
    assert not isfinite(DL2a.core_y)
    assert not DL2a.is_valid
    assert not isfinite(DL2a.average_intensity)

    # Now check that if energy reconstruction is enabled we get a TODO error
    config.ShowerProcessor.reconstruct_energy = True
    process_shower = ShowerProcessor(
        config=config,
        subarray=example_subarray,
        reconstruct_energy=True,
        classify=False
    )
    with pytest.raises(NotImplementedError) as error_info:
        process_shower(example_event)

    # same for classification
    config.ShowerProcessor.classify = True
    process_shower = ShowerProcessor(
        config=config,
        subarray=example_subarray,
        reconstruct_energy=False,
        classify=True
    )
    with pytest.raises(NotImplementedError) as error_info:
        process_shower(example_event)
