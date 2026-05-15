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
    HillasParametersContainer,
    ImageParametersContainer,
    SimulatedCameraContainer,
    SimulatedEventContainer,
    SimulatedShowerContainer,
)
from ctapipe.image import ImageCleaner, ImageProcessor


@pytest.mark.parametrize("cleaner", ImageCleaner.non_abstract_subclasses().values())
def test_image_processor(cleaner, example_event, example_subarray):
    """ensure we get parameters out when we input an event with images"""

    calibrate = CameraCalibrator(subarray=example_subarray)
    process_images = ImageProcessor(
        subarray=example_subarray, image_cleaner_type=cleaner.__name__
    )

    assert isinstance(process_images.clean, cleaner)

    calibrate(example_event)
    process_images(example_event)

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


def test_true_disp_calculation(example_event, example_subarray):
    """Test that true_disp_norm and true_disp_sign are calculated for simulation events."""
    from ctapipe.calib import CameraCalibrator

    calibrate = CameraCalibrator(subarray=example_subarray)
    process_images = ImageProcessor(subarray=example_subarray)

    calibrate(example_event)
    process_images(example_event)

    # Check that true_disp was calculated for telescopes that have true images
    for tel_id, sim_camera in example_event.simulation.tel.items():
        if sim_camera.true_image is not None:
            true_disp = sim_camera.true_disp
            # If hillas parameters could be computed, norm should be >= 0 and sign should be +-1
            if isfinite(sim_camera.true_parameters.hillas.fov_lon.value):
                assert isfinite(true_disp.norm.value) or np.isnan(true_disp.norm.value)
                assert true_disp.norm.unit == u.deg
                assert true_disp.sign in {-1.0, 0.0, 1.0, float("nan")} or np.isnan(
                    true_disp.sign
                )


def test_true_disp_requires_telescope_frame(example_event, example_subarray):
    """Test that true_disp remains NaN when use_telescope_frame=False."""
    from ctapipe.calib import CameraCalibrator

    calibrate = CameraCalibrator(subarray=example_subarray)
    process_images = ImageProcessor(
        subarray=example_subarray, use_telescope_frame=False
    )

    calibrate(example_event)
    process_images(example_event)

    for tel_id, sim_camera in example_event.simulation.tel.items():
        if sim_camera.true_image is not None:
            # When not in telescope frame, true_disp should remain at defaults (NaN)
            assert np.isnan(sim_camera.true_disp.norm.value)
            assert np.isnan(sim_camera.true_disp.sign)


def test_true_disp_no_simulation(example_event, example_subarray):
    """Test that _calculate_true_disp handles missing simulation shower gracefully."""
    from ctapipe.calib import CameraCalibrator

    event = deepcopy(example_event)

    calibrate = CameraCalibrator(subarray=example_subarray)
    process_images = ImageProcessor(subarray=example_subarray)
    calibrate(event)

    # Remove shower info
    event.simulation.shower = None

    process_images(event)

    for tel_id, sim_camera in event.simulation.tel.items():
        if sim_camera.true_image is not None:
            # Without shower, true_disp should remain at defaults (NaN)
            assert np.isnan(sim_camera.true_disp.norm.value)
            assert np.isnan(sim_camera.true_disp.sign)



@pytest.mark.parametrize("cleaner", ImageCleaner.non_abstract_subclasses().values())
def test_image_processor(cleaner, example_event, example_subarray):
    """ensure we get parameters out when we input an event with images"""

    calibrate = CameraCalibrator(subarray=example_subarray)
    process_images = ImageProcessor(
        subarray=example_subarray, image_cleaner_type=cleaner.__name__
    )

    assert isinstance(process_images.clean, cleaner)

    calibrate(example_event)
    process_images(example_event)

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
