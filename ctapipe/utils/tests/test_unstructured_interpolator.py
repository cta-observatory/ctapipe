from ctapipe.utils.unstructured_interpolator import UnstructuredInterpolator
import numpy as np
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
import numpy.ma as ma

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

    interpolated_point = interpolator([[0., 0.5], [0.5, 0.5], [1, 0.5]])
    assert np.all(interpolated_point == [0, 0.5, 1])


def test_linear_nd():
    """
    In its simplest configuration this code should behave exactly the same as the scipy
    LinearNDInterpolator, so lets test that
    """

    # First set up 4 grid points and fill them randomly
    interpolation_points = {(0, 0): np.random.rand(2, 2),
                            (0, 1): np.random.rand(2, 2),
                            (1, 0): np.random.rand(2, 2),
                            (1, 1): np.random.rand(2, 2)}

    # Create UnstructuredInterpolator and LinearNDInterpolator with these points
    interpolator = UnstructuredInterpolator(interpolation_points)
    linear_nd = LinearNDInterpolator(list(interpolation_points.keys()),
                                     list(interpolation_points.values()))

    # Create some random coordinates in this space
    points = np.random.rand(10, 2)
    # And interpolate...
    interpolated_points = interpolator(points)
    linear_nd_points = linear_nd(points)

    # Check everything agrees to a reasonable precision
    assert np.all(np.abs(interpolated_points - linear_nd_points) < 1e-10)


def test_remember_last():
    """
    Check we get the same answer when using masked arrays
    """

    # First set up 4 grid points and fill them randomly
    interpolation_points = {(0, 0): np.random.rand(2, 2),
                            (0, 1): np.random.rand(2, 2),
                            (1, 0): np.random.rand(2, 2),
                            (1, 1): np.random.rand(2, 2)}

    # Create UnstructuredInterpolator and LinearNDInterpolator with these points
    interpolator = UnstructuredInterpolator(interpolation_points, remember_last=True)

    # Create some random coordinates in this space
    random_nums = np.random.rand(2, 2)
    points_mask = ma.masked_array(random_nums, mask=[[True, False],
                                                     [True, False]])

    # And interpolate...
    interpolated_points = interpolator(random_nums).T[0]
    interpolated_points_mask = interpolator(points_mask).T[0]

    # Check everything agrees to a reasonable precision
    assert np.all(np.abs(interpolated_points - interpolated_points_mask) < 1e-10)


def test_masked_input():
    """
    Now lets test how well this all works if we pass a masked input
    """

    # First set up 4 grid points and fill them randomly
    interpolation_points = {(0, 0): np.random.rand(2, 2),
                            (0, 1): np.random.rand(2, 2),
                            (1, 0): np.random.rand(2, 2),
                            (1, 1): np.random.rand(2, 2)}

    # Create UnstructuredInterpolator and LinearNDInterpolator with these points
    interpolator = UnstructuredInterpolator(interpolation_points, remember_last=True)
    linear_nd = LinearNDInterpolator(list(interpolation_points.keys()),
                                     list(interpolation_points.values()))

    # Create some random coordinates in this space
    points = np.random.rand(10, 2)
    # And interpolate...
    interpolator(points)
    interpolated_points = interpolator(points)

    linear_nd_points = linear_nd(points)

    # Check everything agrees to a reasonable precision
    assert np.all(np.abs(interpolated_points - linear_nd_points) < 1e-10)


def test_class_output():
    """
    The final test is to use the more useful functionality of interpolating between the
    outputs of a class member function. I will do this by interpolating between a
    number of numpy regulat grid interpolators and comparing to the output of the
    linear nd interpolator. Again this is a crazy use case, but is a good test.
    """

    x = np.linspace(0, 1, 11)
    # Create a bunch of random numbers to interpolate between
    rand_numbers = np.random.rand(4, 11, 11)

    # Create input for UnstructuredInterpolator
    interpolation_points = {(0, 0): RegularGridInterpolator((x, x), rand_numbers[0]),
                            (0, 1): RegularGridInterpolator((x, x), rand_numbers[1]),
                            (1, 0): RegularGridInterpolator((x, x), rand_numbers[2]),
                            (1, 1): RegularGridInterpolator((x, x), rand_numbers[3])}

    # Create some random points to evaluate our interpolators
    pts1 = np.random.rand(1, 2)
    pts2 = np.random.rand(10, 2)

    interpolator = UnstructuredInterpolator(interpolation_points)
    unsort_value = interpolator(pts1, pts2)

    interpolation_points = {(0, 0): rand_numbers[0],
                            (0, 1): rand_numbers[1],
                            (1, 0): rand_numbers[2],
                            (1, 1): rand_numbers[3]}

    # Perform the same operation by interpolating the values of the full numpy array
    linear_nd = LinearNDInterpolator(list(interpolation_points.keys()),
                                     list(interpolation_points.values()))
    array_out = linear_nd(pts1)
    # Then interpolate on this grid
    reg_interpolator = RegularGridInterpolator((x, x), array_out[0])
    lin_nd_val = reg_interpolator(pts2)

    # Check they give the same answer
    assert np.all(np.abs(unsort_value - lin_nd_val) < 1e-10)


def test_out_of_bounds():
    """
    Test function to check that we sensibly extrapolate when handed a point outside of
    the interpolations bounds
    """

    interpolation_points = {(0, 0): 0.,
                            (0, 1): 0.,
                            (1, 0): 1.,
                            (1, 1): 1.}

    interpolator = UnstructuredInterpolator(interpolation_points)

    interpolated_point = interpolator([[0,2],[1,2],[2,2]])
    assert np.all(interpolated_point == [0., 1., 2.])


if __name__ == '__main__':

    test_simple_interpolation()
    test_linear_nd()
    test_class_output()
    test_out_of_bounds()

