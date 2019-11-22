import pytest
import numpy as np
import astropy.units as u

from ctapipe.image.muon import kundu_chaudhuri_circle_fit
from ctapipe.image.muon.fitting import strip_unit_savely


np.random.seed(0)


def test_kundu_chaudhuri():

    num_tests = 10
    center_xs = np.random.uniform(-1000, 1000, num_tests)
    center_ys = np.random.uniform(-1000, 1000, num_tests)
    radii = np.random.uniform(10, 1000, num_tests)

    for center_x, center_y, radius in zip(center_xs, center_ys, radii):

        phi = np.random.uniform(0, 2 * np.pi, 100)
        x = center_x + radius * np.cos(phi)
        y = center_y + radius * np.sin(phi)

        weights = np.ones_like(x)

        fit_radius, fit_x, fit_y = kundu_chaudhuri_circle_fit(x, y, weights)

        assert np.isclose(fit_x, center_x)
        assert np.isclose(fit_y, center_y)
        assert np.isclose(fit_radius, radius)


def test_kundu_chaudhuri_with_units():

    center_x = 0.5 * u.meter
    center_y = 0.5 * u.meter
    radius = 1 * u.meter

    phi = np.random.uniform(0, 2 * np.pi, 100)
    x = center_x + radius * np.cos(phi)
    y = center_y + radius * np.sin(phi)

    weights = np.ones_like(x)

    fit_radius, fit_x, fit_y = kundu_chaudhuri_circle_fit(x, y, weights)

    assert fit_x.unit == center_x.unit
    assert fit_y.unit == center_y.unit
    assert fit_radius.unit == radius.unit

def test_strip_unit_savely():
    x_m = np.arange(5) * u.m
    y_mm = np.arange(5) * 1000 * u.mm
    z_km = np.arange(5) * 1e-3 * u.km
    nono_deg = np.arange(5) * 1000 * u.deg

    # one argument
    x, unit = strip_unit_savely(x_m)
    assert u.isclose(x, np.arange(5)).all()
    assert (x == np.arange(5)).all()  # wow even this works
    assert unit == u.m

    # two arguments
    x, y, unit = strip_unit_savely(x_m, y_mm)
    assert u.isclose(x, np.arange(5)).all()
    assert (x == np.arange(5)).all()  # wow even this works
    assert unit == u.m
    assert u.isclose(y, np.arange(5)).all()
    assert (y == np.arange(5)).all()  # wow even this works

    x, y, z, unit = strip_unit_savely(x_m, y_mm, z_km)
    assert u.isclose(x, np.arange(5)).all()
    assert (x == np.arange(5)).all()  # wow even this works
    assert unit == u.m
    assert u.isclose(y, np.arange(5)).all()
    assert (y == np.arange(5)).all()  # wow even this works
    assert u.isclose(z, np.arange(5)).all()
    assert (z == np.arange(5)).all()  # wow even this works

    with pytest.raises(u.UnitConversionError):
        strip_unit_savely(x_m, nono_deg)
