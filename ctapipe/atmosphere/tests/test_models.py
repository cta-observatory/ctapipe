import astropy.units as u
import numpy as np
import pytest
from ctapipe.atmosphere import model


@pytest.mark.parametrize(
    "density_model",
    [model.ExponentialAtmosphereDensityProfile()],
)
def test_models(density_model):
    """check that a set of model classes work"""

    # test we can convert to correct units
    density_model(10 * u.km).to(u.kg / u.m**3)

    # check we can also compute the integral
    column_density = density_model.integral(10 * u.km)
    assert column_density.unit.is_equivalent(u.g / u.cm**2)


def test_exponential_model():
    """check exponential models"""

    density_model = model.ExponentialAtmosphereDensityProfile(
        h0=10 * u.m, rho0=0.00125 * u.g / u.cm**3
    )
    assert np.isclose(density_model(1_000_000 * u.km), 0 * u.g / u.cm**3)
    assert np.isclose(density_model(0 * u.km), density_model.rho0)
