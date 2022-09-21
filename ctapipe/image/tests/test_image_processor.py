"""
Tests for ImageProcessor functionality
"""
from copy import deepcopy

import astropy.units as u
import numpy as np
from numpy import isfinite

from ctapipe.calib import CameraCalibrator
from ctapipe.containers import (
    CameraHillasParametersContainer,
    CameraTimingParametersContainer,
)
from ctapipe.image import ImageProcessor
from ctapipe.image.cleaning import MARSImageCleaner


def test_image_processor(example_event, example_subarray):
    """ensure we get parameters out when we input an event with images"""

    calibrate = CameraCalibrator(subarray=example_subarray)
    process_images = ImageProcessor(
        subarray=example_subarray, image_cleaner_type="MARSImageCleaner"
    )

    assert isinstance(process_images.clean, MARSImageCleaner)

    calibrate(example_event)
    process_images(example_event)

    for dl1tel in example_event.dl1.tel.values():
        assert isfinite(dl1tel.image_mask.sum())
        assert isfinite(dl1tel.parameters.hillas.length.value)
        dl1tel.parameters.hillas.length.to("deg")
        assert isfinite(dl1tel.parameters.timing.slope.value)
        assert isfinite(dl1tel.parameters.leakage.pixels_width_1)
        assert isfinite(dl1tel.parameters.concentration.cog)
        assert isfinite(dl1tel.parameters.morphology.n_pixels)
        assert isfinite(dl1tel.parameters.intensity_statistics.max)
        assert isfinite(dl1tel.parameters.peak_time_statistics.max)

    process_images.check_image.to_table()


def test_image_processor_camera_frame(example_event, example_subarray):
    """ensure we get parameters in the camera frame if explicitly specified"""
    event = deepcopy(example_event)

    calibrate = CameraCalibrator(subarray=example_subarray)
    process_images = ImageProcessor(
        subarray=example_subarray,
        use_telescope_frame=False,
        image_cleaner_type="MARSImageCleaner",
    )

    assert isinstance(process_images.clean, MARSImageCleaner)

    calibrate(event)
    process_images(event)

    for dl1tel in event.dl1.tel.values():
        assert isfinite(dl1tel.image_mask.sum())
        assert isfinite(dl1tel.parameters.hillas.length.value)
        dl1tel.parameters.hillas.length.to("meter")
        assert isfinite(dl1tel.parameters.timing.slope.value)
        assert isfinite(dl1tel.parameters.leakage.pixels_width_1)
        assert isfinite(dl1tel.parameters.concentration.cog)
        assert isfinite(dl1tel.parameters.morphology.n_pixels)
        assert isfinite(dl1tel.parameters.intensity_statistics.max)
        assert isfinite(dl1tel.parameters.peak_time_statistics.max)

    process_images.check_image.to_table()

    # set image to zeros to test invalid hillas parameters
    # are in the correct frame
    event = deepcopy(example_event)
    calibrate(event)
    for dl1 in event.dl1.tel.values():
        dl1.image = np.zeros_like(dl1.image)

    process_images(event)
    for dl1 in event.dl1.tel.values():
        assert isinstance(dl1.parameters.hillas, CameraHillasParametersContainer)
        assert isinstance(dl1.parameters.timing, CameraTimingParametersContainer)
        assert np.isnan(dl1.parameters.hillas.length.value)
        assert dl1.parameters.hillas.length.unit == u.m
