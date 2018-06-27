from ctapipe.coordinates.pointing_corrections import PointingCorrection, \
    HESSStylePointingCorrection
import numpy as np


def test_dummy():
    """
    First test is simple, the base class should return a unit diagonal matrix,
    so check this does no modify result when applied
    """
    point = PointingCorrection()
    x, y = np.ones(5), np.zeros(5)
    coordinates = np.matrix([x, y, np.zeros_like(x)])

    transform = point.get_camera_trans_matrix()
    print(transform)

    trans_coords = transform * coordinates

    assert np.all(coordinates == trans_coords)


def test_trans_HESS():
    """
    Test coordinate translation in HESS corrections
    """
    point = HESSStylePointingCorrection(x_trans=1, y_trans=1, rotation=0, scale=1)
    x, y = np.ones(5), np.zeros(5)
    coordinates = np.matrix([x, y, np.ones_like(x)])

    transform = point.get_camera_trans_matrix()

    trans_coords = transform * coordinates
    coordinates_check = np.matrix([x+1, y+1, np.ones_like(x)])

    assert np.all(coordinates_check == trans_coords)


def test_scale_HESS():
    """
    Test coordinate scaling in HESS corrections
    """
    point = HESSStylePointingCorrection(x_trans=0, y_trans=0, rotation=0, scale=2)
    x, y = np.ones(5), np.zeros(5)
    coordinates = np.matrix([x, y, np.ones_like(x)])

    transform = point.get_camera_trans_matrix()

    trans_coords = transform * coordinates
    coordinates_check = np.matrix([x*2, y*2, np.ones_like(x)])

    assert np.sum(coordinates_check - trans_coords) < 1e-10


def test_rotate_HESS():
    """
    Test coordinate rotation in HESS corrections
    """
    point = HESSStylePointingCorrection(x_trans=0, y_trans=0, rotation=np.pi/2, scale=1)
    x, y = np.ones(5), np.zeros(5)
    coordinates = np.matrix([x, y, np.ones_like(x)])

    transform = point.get_camera_trans_matrix()

    trans_coords = transform * coordinates
    coordinates_check = np.matrix([y, x, np.ones_like(x)])

    assert np.sum(coordinates_check - trans_coords) < 1e-10


def test_all_HESS():
    """
    Test all coordinate transforms in HESS corrections
    """
    point = HESSStylePointingCorrection(x_trans=1, y_trans=1, rotation=np.pi/2, scale=3)
    x, y = np.ones(5), np.zeros(5)
    coordinates = np.matrix([x, y, np.ones_like(x)])

    transform = point.get_camera_trans_matrix()
    trans_coords = transform * coordinates
    coordinates_check = np.matrix([(y*3)+1, (x*3)+1, np.ones_like(x)])

    assert np.sum(coordinates_check - trans_coords) < 1e-10