"""
Tests for ImageProcessor functionality
"""

from copy import deepcopy

import astropy.units as u
import numpy as np
import pytest
from numpy import isfinite

from ctapipe.calib import CameraCalibrator
from ctapipe.containers import (
    CameraHillasParametersContainer,
    CameraTimingParametersContainer,
)
from ctapipe.image import ImageCleaner, ImageProcessor


@pytest.mark.parametrize("cleaner", ImageCleaner.non_abstract_subclasses().values())
@pytest.mark.parametrize("dl1_only", [False, True])
def test_image_processor(cleaner, dl1_only, example_event, example_subarray):
    """ensure we get parameters out when we input an event with images"""
    example_event = deepcopy(example_event)

    calibrate = CameraCalibrator(subarray=example_subarray)
    process_images = ImageProcessor(
        subarray=example_subarray, image_cleaner_type=cleaner.__name__
    )

    assert isinstance(process_images.clean, cleaner)

    calibrate(example_event)
    if cleaner.__name__ == "NSBImageCleaner":
        for tel_id, dl1_camera in example_event.dl1.tel.items():
            n_channels = example_subarray.tel[tel_id].camera.readout.n_channels
            pedestal_std = np.ones((n_channels, dl1_camera.image.shape[-1]))
            if n_channels == 2:
                pedestal_std[1] = 1000.0
            monitoring = example_event.monitoring.tel[tel_id].camera
            monitoring.pixel_statistics.pedestal_image.std = pedestal_std
    if dl1_only:
        example_event.r1.tel.clear()
        example_event.dl0.tel.clear()
    process_images(example_event)
    if dl1_only:
        assert not example_event.r1.tel
        assert not example_event.dl0.tel

    for dl1tel in example_event.dl1.tel.values():
        n_survived_pixels = dl1tel.image_mask.sum()
        assert isfinite(n_survived_pixels)
        if n_survived_pixels > 1:
            assert isfinite(dl1tel.parameters.hillas.length.value)
            dl1tel.parameters.hillas.length.to("deg")
            assert isfinite(dl1tel.parameters.timing.slope.value)
            assert isfinite(dl1tel.parameters.leakage.pixels_width_1)
            assert isfinite(dl1tel.parameters.concentration.cog)
            assert isfinite(dl1tel.parameters.morphology.n_pixels)
            assert isfinite(dl1tel.parameters.intensity_statistics.max)
            assert isfinite(dl1tel.parameters.peak_time_statistics.max)
        else:
            assert np.isnan(dl1tel.parameters.hillas.length.value)

    process_images.check_image.to_table()


@pytest.mark.parametrize("cleaner", ImageCleaner.non_abstract_subclasses().values())
def test_image_processor_camera_frame(cleaner, example_event, example_subarray):
    """ensure we get parameters in the camera frame if explicitly specified"""
    event = deepcopy(example_event)

    calibrate = CameraCalibrator(subarray=example_subarray)
    process_images = ImageProcessor(
        subarray=example_subarray,
        use_telescope_frame=False,
        image_cleaner_type=cleaner.__name__,
    )

    assert isinstance(process_images.clean, cleaner)

    calibrate(event)
    process_images(event)

    for dl1tel in event.dl1.tel.values():
        n_survived_pixels = dl1tel.image_mask.sum()
        assert isfinite(n_survived_pixels)
        if n_survived_pixels > 1:
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
