"""
Tests for ImageProcessor functionality
"""

from copy import deepcopy

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import EarthLocation
from numpy import isfinite

from ctapipe.calib import CameraCalibrator
from ctapipe.containers import (
    ArrayEventContainer,
    CameraHillasParametersContainer,
    CameraTimingParametersContainer,
    DL1CameraContainer,
    SimulatedCameraContainer,
    SimulatedEventContainer,
    SimulatedShowerContainer,
)
from ctapipe.coordinates import CameraFrame
from ctapipe.image import ImageCleaner, ImageProcessor
from ctapipe.instrument import CameraGeometry, SubarrayDescription, TelescopeDescription
from ctapipe.instrument.camera import CameraDescription, CameraReadout
from ctapipe.instrument.optics import OpticsDescription, ReflectorShape, SizeType


@pytest.fixture(scope="module")
def synthetic_subarray():
    """
    Create a fully synthetic subarray (no network access needed).
    Uses a rectangular camera with a 1m focal length.
    """
    focal_length = 1.0 * u.m
    geom = CameraGeometry.make_rectangular(10, 10)
    geom.frame = CameraFrame(focal_length=focal_length)
    n_pix = geom.n_pixels

    readout = CameraReadout(
        name="TestCam",
        sampling_rate=1 * u.GHz,
        reference_pulse_shape=np.ones((1, 3)),
        reference_pulse_sample_width=1 * u.ns,
        n_channels=1,
        n_pixels=n_pix,
        n_samples=50,
    )
    cam_desc = CameraDescription("TestCam", geom, readout)
    optics = OpticsDescription(
        name="testoptics",
        size_type=SizeType.MST,
        n_mirrors=1,
        n_mirror_tiles=1,
        mirror_area=1.0 * u.m**2,
        equivalent_focal_length=focal_length,
        effective_focal_length=focal_length,
        reflector_shape=ReflectorShape.PARABOLIC,
    )
    tel_desc = TelescopeDescription(name="TestTel", optics=optics, camera=cam_desc)
    reference_location = EarthLocation(lon=-17 * u.deg, lat=28 * u.deg, height=2200 * u.m)
    return SubarrayDescription(
        name="test",
        tel_positions={1: np.array([0.0, 0.0, 0.0]) * u.m},
        tel_descriptions={1: tel_desc},
        reference_location=reference_location,
    )


@pytest.fixture
def synthetic_event_with_simulation(synthetic_subarray):
    """
    Create a synthetic ArrayEventContainer with simulation data for tel_id=1.
    The image has a simple signal in the central pixels so that Hillas parameters
    can be computed.
    """
    n_pix = synthetic_subarray.tel[1].camera.geometry.n_pixels
    image = np.zeros(n_pix)
    # Put signal in the central pixels (rows 3-7, cols 3-7 of the 10x10 grid)
    # Signal falls off in one direction to give a clear major axis
    for i in range(5):
        image[30 + i * 10 + 3 : 30 + i * 10 + 7] = 100.0 * np.exp(-(i - 2) ** 2 / 2)

    true_image = (image * 0.5).astype(np.int32)

    event = ArrayEventContainer()
    event.monitoring.pointing.array_altitude = 70.0 * u.deg
    event.monitoring.pointing.array_azimuth = 0.0 * u.deg

    event.simulation = SimulatedEventContainer()
    event.simulation.shower = SimulatedShowerContainer(
        alt=70.5 * u.deg,
        az=0.5 * u.deg,
    )

    event.dl1.tel[1] = DL1CameraContainer(image=image, peak_time=None)
    event.monitoring.tel[1]  # initialize monitoring
    event.simulation.tel[1] = SimulatedCameraContainer(true_image=true_image)
    return event


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


def test_true_disp_calculation(
    synthetic_subarray, synthetic_event_with_simulation
):
    """Test that true_disp_norm and true_disp_sign are calculated for simulation events."""
    process_images = ImageProcessor(subarray=synthetic_subarray)
    process_images(synthetic_event_with_simulation)

    sim_camera = synthetic_event_with_simulation.simulation.tel[1]
    true_disp = sim_camera.true_disp

    # Hillas parameters should have been computed for the non-trivial image
    if isfinite(sim_camera.true_parameters.hillas.fov_lon.value):
        assert isfinite(true_disp.norm.value)
        assert true_disp.norm >= 0 * u.deg
        assert true_disp.norm.unit == u.deg
        assert true_disp.sign in {-1.0, 0.0, 1.0}


def test_true_disp_requires_telescope_frame(
    synthetic_subarray, synthetic_event_with_simulation
):
    """Test that true_disp remains NaN when use_telescope_frame=False."""
    process_images = ImageProcessor(
        subarray=synthetic_subarray, use_telescope_frame=False
    )
    process_images(synthetic_event_with_simulation)

    sim_camera = synthetic_event_with_simulation.simulation.tel[1]
    # When not in telescope frame, true_disp should remain at defaults (NaN)
    assert np.isnan(sim_camera.true_disp.norm.value)
    assert np.isnan(sim_camera.true_disp.sign)


def test_true_disp_no_simulation(synthetic_subarray, synthetic_event_with_simulation):
    """Test that _calculate_true_disp handles missing simulation shower gracefully."""
    event = deepcopy(synthetic_event_with_simulation)
    event.simulation.shower = None

    process_images = ImageProcessor(subarray=synthetic_subarray)
    process_images(event)

    sim_camera = event.simulation.tel[1]
    # Without shower, true_disp should remain at defaults (NaN)
    assert np.isnan(sim_camera.true_disp.norm.value)
    assert np.isnan(sim_camera.true_disp.sign)

