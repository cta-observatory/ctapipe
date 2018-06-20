import numpy as np
import astropy.units as u

from ctapipe.image.muon import kundu_chaudhuri_circle_fit

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

