"""tests for ctapipe.utils.coordinates"""

import numpy as np
from astropy import units as u
from numpy.testing import assert_allclose

from ctapipe.utils.coordinates import cartesian_to_polar, polar_to_cartesian


def test_basic_conversion():
    """Test basic cartesian to polar and back conversion."""
    x = 3.0 * u.m
    y = 4.0 * u.m

    # Test cartesian_to_polar
    rho, phi = cartesian_to_polar(x, y)
    assert_allclose(rho.to_value(u.m), 5.0, rtol=1e-6)
    assert_allclose(phi.to_value(u.rad), np.arctan2(4, 3), rtol=1e-6)

    # Test polar_to_cartesian
    x_back, y_back = polar_to_cartesian(rho, phi)
    assert_allclose(x_back.to_value(u.m), 3.0, rtol=1e-6)
    assert_allclose(y_back.to_value(u.m), 4.0, rtol=1e-6)


def test_wrap_angle():
    """Test wrap_angle parameter in cartesian_to_polar."""
    x = -1.0 * u.m
    y = -1.0 * u.m

    _, phi_wrap = cartesian_to_polar(x, y, wrap_angle=True)
    assert_allclose(phi_wrap.to_value(u.rad), 5 * np.pi / 4, rtol=1e-6)
    _, phi_nowrap = cartesian_to_polar(x, y, wrap_angle=False)
    assert_allclose(phi_nowrap.to_value(u.rad), -3 * np.pi / 4, rtol=1e-6)
