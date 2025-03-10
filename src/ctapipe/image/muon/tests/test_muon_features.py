import math

import astropy.units as u
import numpy as np

from ctapipe.containers import MuonRingContainer
from ctapipe.image.muon.features import (
    radial_light_distribution,
    ring_completeness,
    ring_containment,
    ring_size_parameters,
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


def test_ring_size_parameters():
    ring = MuonRingContainer(
        radius=1 * u.deg,
        center_fov_lon=0 * u.deg,
        center_fov_lat=0 * u.deg,
    )
    pixel_x = np.linspace(-2, 2, 1855) * u.deg
    pixel_y = np.linspace(-2, 2, 1855) * u.deg
    ring_integration_width = 0.25
    outer_ring_width = 0.2
    image = np.random.normal(loc=100, scale=10, size=1855)
    image_mask = np.random.choice([True, False], size=1855)

    (
        ring_size,
        size_outside,
        num_pixels_in_ring,
        mean_pixel_outside_ring,
    ) = ring_size_parameters(
        ring,
        pixel_x,
        pixel_y,
        ring_integration_width,
        outer_ring_width,
        image,
        image_mask,
    )

    assert ring_size > 0
    assert size_outside > 0
    assert num_pixels_in_ring > 0
    assert mean_pixel_outside_ring > 0


def test_radial_light_distribution():
    center_x = 0.0 * u.deg
    center_y = 0.0 * u.deg
    pixel_x = np.linspace(-10, 10, 1855) * u.deg
    pixel_y = np.linspace(-10, 10, 1855) * u.deg
    image = np.random.normal(loc=100, scale=10, size=1855)

    standard_dev, skewness, excess_kurtosis = radial_light_distribution(
        center_x, center_y, pixel_x, pixel_y, image
    )

    assert standard_dev.unit == u.deg
    assert np.isfinite(standard_dev.value)
    assert np.isfinite(skewness)
    assert np.isfinite(excess_kurtosis)


def test_radial_light_distribution_zero_image():
    center_x = 0.0 * u.deg
    center_y = 0.0 * u.deg
    pixel_x = np.linspace(-10, 10, 1855) * u.deg
    pixel_y = np.linspace(-10, 10, 1855) * u.deg
    image = np.zeros(1855)

    standard_dev, skewness, excess_kurtosis = radial_light_distribution(
        center_x, center_y, pixel_x, pixel_y, image
    )

    assert np.isnan(standard_dev)
    assert np.isnan(skewness)
    assert np.isnan(excess_kurtosis)
