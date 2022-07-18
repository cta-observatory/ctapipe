import astropy.units as u
import numpy as np
import pytest

from ctapipe.atmosphere import model
from ctapipe.utils import get_dataset_path

SIMTEL_PATH = get_dataset_path(
    "gamma_20deg_0deg_run2___cta-prod5-paranal_desert-2147m-Paranal-dark_cone10-100evts.simtel.zst"
)


@pytest.fixture(scope="session")
def table_profile():
    return get_simtel_profile_from_eventsource()


def get_simtel_profile_from_eventsource():
    """get a TableAtmosphereDensityModel from a simtel file"""
    from ctapipe.io import EventSource

    with EventSource(SIMTEL_PATH) as source:
        return source.atmosphere_density_profile


def get_simtel_fivelayer_profile():
    from ctapipe.io.simteleventsource import read_atmosphere_profile_from_simtel

    return read_atmosphere_profile_from_simtel(SIMTEL_PATH, kind="fivelayer")


@pytest.mark.parametrize(
    "density_model",
    [
        model.ExponentialAtmosphereDensityProfile(),
        get_simtel_profile_from_eventsource(),
        get_simtel_fivelayer_profile(),
    ],
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


def test_table_model_interpolation(table_profile):
    """check that interpolation is reasonable"""

    np.testing.assert_allclose(
        table_profile(table_profile.table["height"].to("km")),
        table_profile.table["density"].to("g cm-3"),
    )

    # check that fine interpolation up to 100 km :
    height_fine = np.linspace(0, 100, 1000) * u.km
    assert np.isfinite(table_profile.integral(height_fine)).all()
