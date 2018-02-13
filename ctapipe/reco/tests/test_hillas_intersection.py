from ctapipe.reco.hillas_intersection import HillasIntersection
import astropy.units as u
from numpy.testing import assert_allclose
import numpy as np


def test_intersect():
    """
    Simple test to check the intersection of lines. Try to intersect positions at (0,1) and (1,0)
    with angles perpendicular and test they cross at (0,0)
    """
    hill = HillasIntersection()
    x1 = 0
    y1 = 1
    theta1 = 90 * u.deg

    x2 = 1
    y2 = 0
    theta2 = 0 * u.deg

    sx, sy = hill.intersect_lines(x1, y1, theta1, x2, y2, theta2)

    assert_allclose(sx, 0, atol=1e-6)
    assert_allclose(sy, 0, atol=1e-6)


def test_parallel():
    """
    Simple test to check the intersection of lines. Try to intersect positions at (0,0) and (0,1)
    with angles parallel and check the behaviour
    """
    hill = HillasIntersection()
    x1 = 0
    y1 = 0
    theta1 = 0 * u.deg

    x2 = 1
    y2 = 0
    theta2 = 0 * u.deg

    sx, sy = hill.intersect_lines(x1, y1, theta1, x2, y2, theta2)
    assert_allclose(sx, np.nan, atol=1e-6)
    assert_allclose(sy, np.nan, atol=1e-6)

test_intersect()
test_parallel()
