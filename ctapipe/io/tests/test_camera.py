from ..camera import *


def test_make_rectangular_camera_geometry():
    geom = make_rectangular_camera_geometry()
    assert(geom.pix_x.shape == geom.pix_y.shape)


def test_load_hess_camera():
    geom = get_camera_geometry("hess", 1)
