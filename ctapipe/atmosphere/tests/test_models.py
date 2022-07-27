import astropy.units as u
import numpy as np
import pytest

from ctapipe.atmosphere import model
from ctapipe.utils import get_dataset_path

SIMTEL_PATH = get_dataset_path(
    "gamma_20deg_0deg_run2___cta-prod5-paranal_desert-2147m-Paranal-dark_cone10-100evts.simtel.zst"
)


def get_model_from_simtel():
    """get a TableAtmosphereDensityModel from a simtel file"""
    from ctapipe.io import EventSource

    with EventSource(SIMTEL_PATH) as source:
        return source.atmosphere_density_profiles[0]


@pytest.mark.parametrize(
    "density_model",
    [model.ExponentialAtmosphereDensityProfile(), get_model_from_simtel()],
)
def test_models(density_model):
    """check that a set of model classes work"""

    # test we can convert to correct units
    density_model(10 * u.km).to(u.kg / u.m**3)

    # ensure units are properly taken into account
    assert np.isclose(density_model(1 * u.km), density_model(1000 * u.m))

    # check we can also compute the integral
    column_density = density_model.integral(10 * u.km)
    assert column_density.unit.is_equivalent(u.g / u.cm**2)

    assert np.isclose(
        density_model.integral(1 * u.km), density_model.integral(1000 * u.m)
    )


def test_exponential_model():
    """check exponential models"""

    density_model = model.ExponentialAtmosphereDensityProfile(
        h0=10 * u.m, rho0=0.00125 * u.g / u.cm**3
    )
    assert np.isclose(density_model(1_000_000 * u.km), 0 * u.g / u.cm**3)
    assert np.isclose(density_model(0 * u.km), density_model.rho0)
