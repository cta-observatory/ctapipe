import numpy as np
import astropy.units as u

from ctapipe.image.muon.features import ring_containment

def test_ring_containment():
    ring_radius = 1. * u.m
    cam_radius = 2.25 * u.m
    ring_center_x = 1.5 * u.m
    ring_center_y = 1.5 * u.m

    ring_cont = ring_containment(
        ring_radius, cam_radius,
        ring_center_x, ring_center_y)

    assert(ring_cont <= 1. and ring_cont >= 0.)

if __name__ == '__main__':
    test_ring_containment()
