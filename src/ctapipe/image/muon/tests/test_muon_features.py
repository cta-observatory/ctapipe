import math

import astropy.units as u
import numpy as np

from ctapipe.image.muon.features import ring_completeness, ring_containment


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
