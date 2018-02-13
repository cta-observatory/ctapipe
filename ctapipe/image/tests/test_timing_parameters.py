from ctapipe.image.timing_parameters import timing_parameters
import numpy as np
import astropy.units as u
from numpy.testing import assert_allclose


def test_grad_fit():
    """
    Simple test that gradient fitting gives expected answers for perfect
    gradient
    """
    grad = 2
    intercept = 1

    timing = timing_parameters(
        pix_x=np.zeros(4) * u.deg,
        pix_y=np.arange(4) * u.deg,
        image=np.ones(4),
        peak_time=intercept * u.ns + grad * np.arange(4) * u.ns,
        rotation_angle=0 * u.deg
    )

    # Test we get the values we put in back out again
    assert_allclose(timing.gradient, grad * u.ns / u.deg)
    assert_allclose(timing.intercept, intercept * u.deg)

    # Then try a different rotation angle
    rot_angle = 20 * u.deg
    timing_rot20 = timing_parameters(
        pix_x=np.zeros(4) * u.deg,
        pix_y=np.arange(4) * u.deg,
        image=np.ones(4),
        peak_time=intercept * u.ns +
        grad * np.arange(4) * u.ns,
        rotation_angle=rot_angle
    )
    # Test the output again makes sense
    assert_allclose(timing_rot20.gradient, timing.gradient / np.cos(rot_angle))
    assert_allclose(timing_rot20.intercept, timing.intercept)
