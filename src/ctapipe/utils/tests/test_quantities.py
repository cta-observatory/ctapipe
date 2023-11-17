"""tests for ctapipe.utils.quantities"""
import astropy.units as u
import numpy as np
import pytest

from ctapipe.utils.quantities import all_to_value


def test_all_to_value():
    """test all_to_value"""
    x_m = np.arange(5) * u.m
    y_mm = np.arange(5) * 1000 * u.mm
    z_km = np.arange(5) * 1e-3 * u.km
    nono_deg = np.arange(5) * 1000 * u.deg

    # one argument
    x = all_to_value(x_m, unit=u.m)
    assert (x == np.arange(5)).all()

    # two arguments
    x, y = all_to_value(x_m, y_mm, unit=u.m)
    assert (x == np.arange(5)).all()
    assert (y == np.arange(5)).all()

    # three
    x, y, z = all_to_value(x_m, y_mm, z_km, unit=u.m)
    assert (x == np.arange(5)).all()
    assert (y == np.arange(5)).all()
    assert (z == np.arange(5)).all()

    # cannot be converted
    with pytest.raises(u.UnitConversionError):
        all_to_value(x_m, nono_deg, unit=x_m.unit)
