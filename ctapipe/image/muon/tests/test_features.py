import astropy.units as u
import numpy as np
import math
from ctapipe.image.muon.features import ring_containment
from ctapipe.image.muon.features import ring_completeness
from ctapipe.image.muon.features import npix_above_threshold
from ctapipe.image.muon.features import npix_composing_ring

def test_ring_containment():

    ring_radius = 1. * u.m
    cam_radius = 2.25 * u.m
    ring_center_x = 1.5 * u.m
    ring_center_y = 1.5 * u.m

    ring_cont = ring_containment(
        ring_radius, cam_radius,
        ring_center_x, ring_center_y)

    assert(ring_cont <= 1. and ring_cont >= 0.)


def test_ring_completeness():

    angle_ring = np.linspace(0, 2 * math.pi, 360.)
    x = np.cos(angle_ring) * u.m
    y = np.sin(angle_ring) * u.m
    pe = np.random.uniform(0, 100, len(x))
    ring_radius = 1. * u.m

    ring_center_x = 0. * u.m
    ring_center_y = 0. * u.m

    ring_comp = ring_completeness(
        x, y, pe, ring_radius,
        ring_center_x, ring_center_y,
        30, 30)

    assert(ring_comp <= 1. and ring_comp >= 0.)


def test_npix_above_threshold():
    len_array = 100
    thr = 5.
    pix = np.random.uniform(0, 100, len_array)

    npix = npix_above_threshold(pix, thr)

    assert((npix >= 0) and (npix <= len_array))


def test_npix_composing_ring():
    len_array = 100
    pix = np.random.uniform(0, 100, len_array)

    npix = npix_composing_ring(pix)

    assert((npix >= 0) and (npix <= len_array))
