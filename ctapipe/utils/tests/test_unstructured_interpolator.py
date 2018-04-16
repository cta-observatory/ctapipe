from ctapipe.utils.unstructured_interpolator import UnstructuredInterpolator
import numpy as np
from scipy.interpolate import LinearNDInterpolator


def test_simple_interpolation():
    """
    Simple test function to asses the basic funtionality of the unstructured
    interpolator, check if we just spread points on a grid
    """

    interpolation_points = {(0, 0): 0.,
                            (0, 1): 0.,
                            (1, 0): 1.,
                            (1, 1): 1.}

    interpolator = UnstructuredInterpolator(interpolation_points)

    # OK lets first just check we get the values at the grid points back out again...
    interpolated_point = interpolator([[0, 0], [0, 1],
                                       [1, 0], [1, 1]])
    assert np.all(interpolated_point == [0., 0., 1., 1.])

    interpolated_point = interpolator([[0., 0.5], [0.5, 0.5], [1, 0.5] ])
    assert np.all(interpolated_point == [0, 0.5, 1])


def test_linear_nd():
    """
    In its simplest configuration this code should behave exactly the same as the scipy
    LinearNDInterpolator, so lets test that
    """

    # First set up 4 grid points and fill them randomly
    interpolation_points = {(0, 0): np.random.rand(2,2),
                            (0, 1): np.random.rand(2,2),
                            (1, 0): np.random.rand(2,2),
                            (1, 1): np.random.rand(2,2)}

    # Create UnstructuredInterpolator and LinearNDInterpolator with these points
    interpolator = UnstructuredInterpolator(interpolation_points)
    linear_nd = LinearNDInterpolator(list(interpolation_points.keys()),
                                     list(interpolation_points.values()))

    # Create some random coordinates in this space
    points = np.random.rand(10,2)
    # And interpolate...
    interpolated_points = interpolator(points)
    linear_nd_points = linear_nd(points)

    # Check everything agrees to a reasonable precision
    assert np.all(np.abs(interpolated_points - linear_nd_points) < 1e-10)


if __name__ == '__main__':

    test_simple_interpolation()
    test_linear_nd()
