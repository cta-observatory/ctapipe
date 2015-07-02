from ..camera import *
import numpy as np


def test_make_rectangular_camera_geometry():
    geom = make_rectangular_camera_geometry()
    assert(geom.pix_x.shape == geom.pix_y.shape)


def test_load_hess_camera():
    get_camera_geometry("hess", 1)


def test_find_neighbor_pixels():
    x, y = np.meshgrid(np.linspace(-5, 5, 5), np.linspace(-5, 5, 5))
    neigh = find_neighbor_pixels(x.ravel(), y.ravel(), rad=3.1)
    assert(neigh[11] == [6, 11, 10, 12])
