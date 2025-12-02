import math

import astropy.units as u
import numpy as np

from ctapipe.containers import MuonRingContainer
from ctapipe.image.muon.features import (
    radial_light_distribution,
    ring_completeness,
    ring_containment,
    ring_intensity_parameters,
)


def test_ring_containment():
    ring = MuonRingContainer(
        radius=1 * u.deg,
        center_fov_lon=0 * u.deg,
        center_fov_lat=0 * u.deg,
        center_phi=0 * u.deg,
        center_distance=0 * u.deg,
    )
    cam_radius = 4 * u.deg

    containment = ring_containment(ring, cam_radius)
    assert containment == 1.0

    ring.center_fov_lat = cam_radius
    ring.center_distance = cam_radius
    containment = ring_containment(ring, cam_radius)
    assert 0.4 <= containment <= 0.5

    ring.center_fov_lat = cam_radius + 1.1 * ring.radius
    ring.center_distance = cam_radius + 1.1 * ring.radius
    containment = ring_containment(ring, cam_radius)
    assert containment == 0.0


def test_ring_completeness():
    rng = np.random.default_rng(0)

    angle_ring = np.linspace(0, 2 * math.pi, 360)
    lon = np.cos(angle_ring) * u.deg
    lat = np.sin(angle_ring) * u.deg
    pe = rng.uniform(50, 100, len(lon))
    ring = MuonRingContainer(
        radius=1.0 * u.deg,
        center_fov_lon=0.0 * u.deg,
        center_fov_lat=0.0 * u.deg,
    )

    ring_comp = ring_completeness(lon, lat, pe, ring, threshold=30, bins=30)

    # Since we are generating a complete ring with uniform weights, we expect the completeness to be 1.0
    assert ring_comp == 1.0

    # Test with a partial ring
    partial_pe = np.concatenate([pe[:180], np.zeros(180)])
    ring_comp_partial = ring_completeness(
        lon, lat, partial_pe, ring, threshold=30, bins=30
    )

    # Since half of the ring is missing, we expect the completeness to be around 0.5
    assert 0.4 <= ring_comp_partial <= 0.6


def test_ring_intensity_parameters():
    ring = MuonRingContainer(
        radius=1 * u.deg,
        center_fov_lon=0 * u.deg,
        center_fov_lat=0 * u.deg,
    )
    pixel_fov_lon = np.linspace(-2, 2, 1855) * u.deg
    pixel_fov_lat = np.linspace(-2, 2, 1855) * u.deg
    ring_integration_width = 0.25
    outer_ring_width = 0.2

    # Create a synthetic image with known properties
    image = np.ones(1855)  # Uniform intensity
    image_mask = np.ones(1855, dtype=bool)  # All pixels are valid

    # Calculate expected values
    dist = np.hypot(
        pixel_fov_lon - ring.center_fov_lon, pixel_fov_lat - ring.center_fov_lat
    )
    dist_mask = np.abs(dist - ring.radius) < (ring.radius * ring_integration_width)
    ring_intensity_expected = np.sum(image[dist_mask])
    intensity_outside_ring_expected = np.sum(image[~dist_mask])
    n_pixels_in_ring_expected = np.sum(dist_mask)
    dist_mask_2 = np.logical_and(
        ~dist_mask,
        np.abs(dist - ring.radius)
        < ring.radius * (ring_integration_width + outer_ring_width),
    )
    mean_intensity_outside_ring_expected = np.sum(image[dist_mask_2]) / len(
        image[dist_mask_2]
    )

    (
        ring_intensity,
        intensity_outside_ring,
        n_pixels_in_ring,
        mean_intensity_outside_ring,
    ) = ring_intensity_parameters(
        ring,
        pixel_fov_lon,
        pixel_fov_lat,
        ring_integration_width,
        outer_ring_width,
        image,
        image_mask,
    )

    assert np.isclose(ring_intensity, ring_intensity_expected, rtol=1e-2)
    assert np.isclose(
        intensity_outside_ring, intensity_outside_ring_expected, rtol=1e-2
    )
    assert n_pixels_in_ring == n_pixels_in_ring_expected
    assert np.isclose(
        mean_intensity_outside_ring, mean_intensity_outside_ring_expected, rtol=1e-2
    )


def test_radial_light_distribution():
    center_x = 0.0 * u.deg
    center_y = 0.0 * u.deg
    pixel_x = np.linspace(-2, 2, 1855) * u.deg
    pixel_y = np.linspace(-2, 2, 1855) * u.deg

    # Create a synthetic image with known properties
    image = np.ones(1855)  # Uniform intensity

    # Calculate expected values
    expected_std_dev = (
        np.sqrt(8) / np.sqrt(12) * u.deg
    )  # (b - a) / sqrt(12) where b-a is a diagonal of a square with the side length 4
    expected_skewness = 0.0  # Uniform distribution has zero skewness
    expected_excess_kurtosis = -1.2  # Uniform distribution has excess kurtosis of -1.2

    radial_std_dev, skewness, excess_kurtosis = radial_light_distribution(
        center_x, center_y, pixel_x, pixel_y, image
    )

    assert radial_std_dev.unit == u.deg
    assert isinstance(skewness, (float, np.floating)), (
        f"Unexpected type: {type(skewness)}"
    )
    assert isinstance(excess_kurtosis, (float, np.floating)), (
        f"Unexpected type: {type(excess_kurtosis)}"
    )
    assert np.isclose(radial_std_dev, expected_std_dev, atol=1e-2)
    assert np.isclose(skewness, expected_skewness, atol=1e-2)
    assert np.isclose(excess_kurtosis, expected_excess_kurtosis, atol=1e-2)


def test_radial_light_distribution_zero_image():
    center_x = 0.0 * u.deg
    center_y = 0.0 * u.deg
    pixel_x = np.linspace(-10, 10, 1855) * u.deg
    pixel_y = np.linspace(-10, 10, 1855) * u.deg
    image = np.zeros(1855)

    radial_std_dev, skewness, excess_kurtosis = radial_light_distribution(
        center_x, center_y, pixel_x, pixel_y, image
    )

    assert np.isnan(radial_std_dev)
    assert np.isnan(skewness)
    assert np.isnan(excess_kurtosis)
