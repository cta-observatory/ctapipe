from ..camera import CameraGeometry, make_rectangular_camera_geometry
from ..camera import find_neighbor_pixels
import numpy as np
from astropy import units as u


def test_make_rectangular_camera_geometry():
    geom = make_rectangular_camera_geometry()
    assert(geom.pix_x.shape == geom.pix_y.shape)


def test_load_hess_camera():
    geom = CameraGeometry.from_name("hess", 1)
    assert len(geom.pix_x) == 960


def test_rotate_camera():
    geom = make_rectangular_camera_geometry(10, 10)
    #geom.rotate('10.0d')
    geom.rotate(10* u.deg)


def test_guess_camera():
    px = np.linspace(-10, 10, 11328) * u.m
    py = np.linspace(-10, 10, 11328) * u.m
    geom = CameraGeometry.guess(px, py,0 * u.m)
    assert geom.pix_type.startswith('rect')


def test_find_neighbor_pixels():
    x, y = np.meshgrid(np.linspace(-5, 5, 5), np.linspace(-5, 5, 5))
    neigh = find_neighbor_pixels(x.ravel(), y.ravel(), rad=3.1)
    assert(set(neigh[11]) == set([16, 6, 10, 12]))
