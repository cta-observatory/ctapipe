"""
Checks for the atmosphere module
"""

# pylint: disable=import-outside-toplevel
from unittest.mock import patch

import astropy.units as u
import numpy as np
import pytest

import ctapipe.atmosphere as atmo
from ctapipe.utils import get_dataset_path

SIMTEL_PATH = get_dataset_path(
    "gamma_20deg_0deg_run2___cta-prod5-paranal_desert"
    "-2147m-Paranal-dark_cone10-100evts.simtel.zst"
)


def get_simtel_profile_from_eventsource():
    """get a TableAtmosphereDensityProfile from a simtel file"""
    from ctapipe.io import EventSource

    with EventSource(SIMTEL_PATH) as source:
        return source.atmosphere_density_profile


@pytest.fixture(scope="session")
def table_profile():
    """a table profile for testing"""
    return get_simtel_profile_from_eventsource()


@pytest.fixture(scope="session")
def layer5_profile():
    """a 5 layer profile for testing"""
    fit_reference = np.array(
        [
            [0.00 * 100000, -140.508, 1178.05, 994186, 0],
            [9.75 * 100000, -18.4377, 1265.08, 708915, 0],
            [19.00 * 100000, 0.217565, 1349.22, 636143, 0],
            [46.00 * 100000, -0.000201796, 703.745, 721128, 0],
            [106.00 * 100000, 0.000763128, 1, 1.57247e10, 0],
        ]
    )

    profile_5 = atmo.FiveLayerAtmosphereDensityProfile.from_array(fit_reference)
    return profile_5


def get_simtel_fivelayer_profile():
    """
    get a sample 3-layer profile
    """
    from ctapipe.io.simteleventsource import (
        AtmosphereProfileKind,
        read_atmosphere_profile_from_simtel,
    )

    return read_atmosphere_profile_from_simtel(
        SIMTEL_PATH, kind=AtmosphereProfileKind.FIVELAYER
    )


@pytest.mark.parametrize(
    "density_model",
    [
        atmo.ExponentialAtmosphereDensityProfile(),
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
    assert column_density.unit.is_equivalent(atmo.GRAMMAGE_UNIT)

    assert np.isclose(
        density_model.integral(1 * u.km), density_model.integral(1000 * u.m)
    )

    pytest.importorskip("matplotlib")
    with patch("matplotlib.pyplot.show"):
        density_model.peek()


def test_exponential_model():
    """check exponential models"""

    density_model = atmo.ExponentialAtmosphereDensityProfile(
        scale_height=10 * u.m, scale_density=0.00125 * atmo.DENSITY_UNIT
    )
    assert np.isclose(density_model(1_000_000 * u.km), 0 * atmo.DENSITY_UNIT)
    assert np.isclose(density_model(0 * u.km), density_model.scale_density)


def test_table_model_interpolation(table_profile):
    """check that interpolation is reasonable"""

    np.testing.assert_allclose(
        table_profile(table_profile.table["height"].to("km")),
        table_profile.table["density"].to("g cm-3"),
    )

    # check that fine interpolation up to 100 km :
    height_fine = np.linspace(0, 100, 1000) * u.km
    assert np.isfinite(table_profile.integral(height_fine)).all()


def test_against_reference(layer5_profile):
    """
    Test five-layer and table methods against a reference analysis from
    SimTelArray.  Data from communication with K. Bernloehr.

    See https://github.com/cta-observatory/ctapipe/pull/2000
    """
    from ctapipe.utils import get_table_dataset

    reference_table = get_table_dataset(
        "atmosphere_profile_comparison_from_simtelarray"
    )

    height = reference_table["Altitude_km"].to("km")

    np.testing.assert_allclose(
        1.0 - layer5_profile(height) / reference_table["rho_5"], 0, atol=1e-5
    )
    np.testing.assert_allclose(
        1.0
        - layer5_profile.slant_depth_from_height(height) / reference_table["thick_5"],
        0,
        atol=1e-5,
    )


def test_slant_depth_from_height(table_profile):
    """
    Test observer altitude and zenith angle dependence of LoS integral function
    """

    assert table_profile.slant_depth_from_height(
        20 * u.km
    ) < table_profile.slant_depth_from_height(10 * u.km)

    assert table_profile.slant_depth_from_height(
        10 * u.km, zenith_angle=30 * u.deg
    ) > table_profile.slant_depth_from_height(10 * u.km, zenith_angle=0 * u.deg)

    los_integral_z_0 = table_profile.slant_depth_from_height(
        table_profile.table["height"].to("km")[5:15]
    )

    assert np.allclose(
        los_integral_z_0,
        table_profile.table["column_density"].to("g cm-2")[5:15],
        rtol=1e-3,
    )

    los_integral_z_20 = table_profile.slant_depth_from_height(
        table_profile.table["height"].to("km")[5:15],
        zenith_angle=20 * u.deg,
    )

    assert np.allclose(
        los_integral_z_20,
        table_profile.table["column_density"].to("g cm-2")[5:15]
        / np.cos(np.deg2rad(20)),
        rtol=1e-3,
    )


def test_height_overburden_roundtrip(table_profile, layer5_profile):
    """
    Test that successive application of height to overburden
    and overburden to height functions return original values
    """

    layer_5_heights = u.Quantity([5, 15, 30, 70, 110] * u.km)

    for height in layer_5_heights:
        circle_height_5_layer = layer5_profile.height_from_overburden(
            layer5_profile.integral(height)
        )

        assert np.allclose(circle_height_5_layer, height, rtol=0.005)

    # Exponential atmosphere
    density_model = atmo.ExponentialAtmosphereDensityProfile(
        scale_height=10 * u.km, scale_density=0.00125 * atmo.DENSITY_UNIT
    )

    circle_height_exponential = density_model.height_from_overburden(
        density_model.integral(47 * u.km)
    )

    assert np.allclose(circle_height_exponential, 47 * u.km, rtol=0.0001)

    circle_height_table = table_profile.height_from_overburden(
        table_profile.integral(47 * u.km)
    )

    assert np.allclose(circle_height_table, 47 * u.km, rtol=0.0001)


def test_height_slant_depth_roundtrip(table_profile, layer5_profile):
    """
    Test that successive application of height to overburden
    and overburden to height functions return original values
    """

    layer_5_heights = u.Quantity([5, 15, 30, 70, 110] * u.km)

    for height in layer_5_heights:
        circle_height_5_layer = layer5_profile.height_from_slant_depth(
            layer5_profile.slant_depth_from_height(height)
        )

        assert np.allclose(circle_height_5_layer, height, rtol=0.005)

    for height in layer_5_heights:
        circle_height_5_layer_z_20 = layer5_profile.height_from_slant_depth(
            layer5_profile.slant_depth_from_height(height, zenith_angle=20 * u.deg),
            zenith_angle=20 * u.deg,
        )

        assert np.allclose(circle_height_5_layer_z_20, height, rtol=0.005)

    # Exponential atmosphere
    density_model = atmo.ExponentialAtmosphereDensityProfile(
        scale_height=10 * u.km, scale_density=0.00125 * atmo.DENSITY_UNIT
    )

    circle_height_exponential = density_model.height_from_slant_depth(
        density_model.slant_depth_from_height(47 * u.km)
    )

    assert np.allclose(circle_height_exponential, 47 * u.km, rtol=0.0001)

    circle_height_exponential_z_20 = density_model.height_from_slant_depth(
        density_model.slant_depth_from_height(47 * u.km, zenith_angle=20 * u.deg),
        zenith_angle=20 * u.deg,
    )

    assert np.allclose(circle_height_exponential_z_20, 47 * u.km, rtol=0.0001)

    circle_height_table = table_profile.height_from_slant_depth(
        table_profile.slant_depth_from_height(47 * u.km)
    )

    assert np.allclose(circle_height_table, 47 * u.km, rtol=0.0001)

    circle_height_table_z_20 = table_profile.height_from_slant_depth(
        table_profile.slant_depth_from_height(47 * u.km, zenith_angle=20 * u.deg),
        zenith_angle=20 * u.deg,
    )

    assert np.allclose(circle_height_table_z_20, 47 * u.km, rtol=0.0001)


def test_out_of_range_table(table_profile):
    with pytest.warns(RuntimeWarning, match="divide by zero"):
        assert np.isposinf(table_profile.height_from_overburden(0 * atmo.GRAMMAGE_UNIT))

    assert np.isnan(table_profile.height_from_overburden(2000 * atmo.GRAMMAGE_UNIT))

    assert table_profile(150 * u.km).value == 0.0
    assert np.isnan(table_profile(-0.1 * u.km))

    assert table_profile.integral(150 * u.km).value == 0.0
    assert np.isnan(table_profile.integral(-0.1 * u.km))


def test_out_of_range_exponential():
    density_model = atmo.ExponentialAtmosphereDensityProfile(
        scale_height=10 * u.km, scale_density=0.00125 * atmo.DENSITY_UNIT
    )

    with pytest.warns(RuntimeWarning, match="divide by zero"):
        assert np.isposinf(density_model.height_from_overburden(0 * atmo.GRAMMAGE_UNIT))

    assert np.isnan(density_model.height_from_overburden(2000 * atmo.GRAMMAGE_UNIT))

    assert np.isnan(density_model(-0.1 * u.km))

    assert np.isnan(density_model.integral(-0.1 * u.km))


def test_out_of_range_five_layer(layer5_profile):
    # Five-layer atmosphere

    assert np.isposinf(layer5_profile.height_from_overburden(0 * atmo.GRAMMAGE_UNIT))

    assert np.isnan(layer5_profile.height_from_overburden(2000 * atmo.GRAMMAGE_UNIT))

    assert np.allclose(layer5_profile(150 * u.km).value, 0.0, atol=1e-9)
    assert np.isnan(layer5_profile(-0.1 * u.km))

    assert np.allclose(layer5_profile.integral(150 * u.km).value, 0.0, atol=1e-9)
    assert np.isnan(layer5_profile.integral(-0.1 * u.km))
