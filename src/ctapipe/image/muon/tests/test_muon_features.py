import math

import astropy.units as u
import numpy as np

from ctapipe.image.muon.features import (
    ring_completeness, 
    ring_containment,
    ring_size_parameters, 
    radial_light_distribution
)


def test_ring_containment():
    ring_radius = 1 * u.deg
    cam_radius = 4 * u.deg

    ring_center_x = 0 * u.deg
    ring_center_y = 0 * u.deg
    containment = ring_containment(
        ring_radius, ring_center_x, ring_center_y, cam_radius
    )
    assert containment == 1.0

    ring_center_x = 0 * u.deg
    ring_center_y = cam_radius
    containment = ring_containment(
        ring_radius, ring_center_x, ring_center_y, cam_radius
    )
    assert 0.4 <= containment <= 0.5

    ring_center_x = 0 * u.deg
    ring_center_y = cam_radius + 1.1 * ring_radius
    containment = ring_containment(
        ring_radius, ring_center_x, ring_center_y, cam_radius
    )
    assert containment == 0.0


def test_ring_completeness():
    rng = np.random.default_rng(0)

    angle_ring = np.linspace(0, 2 * math.pi, 360)
    x = np.cos(angle_ring) * u.m
    y = np.sin(angle_ring) * u.m
    pe = rng.uniform(0, 100, len(x))
    ring_radius = 1.0 * u.m

    ring_center_x = 0.0 * u.m
    ring_center_y = 0.0 * u.m

    ring_comp = ring_completeness(
        x, y, pe, ring_radius, ring_center_x, ring_center_y, 30, 30
    )

    assert ring_comp <= 1
    assert ring_comp >= 0


def test_ring_size_parameters():
    radius = 1
    center_x = 0
    center_y = 0
    pixel_x = np.linspace(-2, 2, 1855)
    pixel_y = np.linspace(-2, 2, 1855)
    ring_integration_width = 0.25
    outer_ring_width = 0.2
    image = np.random.normal(loc=100, scale=10, size=1855)
    image_mask = np.random.choice([True, False], size=1855)

    ring_size, size_outside, num_pixels_in_ring, mean_pixel_outside_ring = ring_size_parameters(
        radius,
        center_x,
        center_y,
        pixel_x,
        pixel_y,
        ring_integration_width,
        outer_ring_width,
        image,
        image_mask
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
