"""
This module contains the ctapipe.image.psf_model unit tests
"""

import astropy.units as u
import numpy as np
import pytest

from ctapipe.compat import trapz_func
from ctapipe.instrument.optics import PSFModel


@pytest.fixture(scope="session")
def coma_psf(example_subarray):
    lst_plate_scale = np.rad2deg(
        1.0 / example_subarray.tel[1].optics.equivalent_focal_length.to_value(u.m)
    )

    psf = PSFModel.from_name(
        "ComaPSFModel",
        subarray=example_subarray,
        asymmetry_max=0.5,
        asymmetry_decay_rate=10 / lst_plate_scale,
        asymmetry_linear_term=0.15 / lst_plate_scale,
        radial_scale_offset=0.015 * lst_plate_scale,
        radial_scale_linear=-0.1,
        radial_scale_quadratic=0.06 / lst_plate_scale**1,
        radial_scale_cubic=0.03 / lst_plate_scale**2,
        polar_scale_amplitude=0.25 * lst_plate_scale,
        polar_scale_decay=7.5 / lst_plate_scale,
        polar_scale_offset=0.02 * lst_plate_scale,
    )
    return psf


@pytest.fixture(scope="session")
def zernike_psf(example_subarray):
    return PSFModel.from_name("ZernikePSFModel", subarray=example_subarray)


# Source position and evaluation grids used by all tests below. These are
# identical for every PSF model; only the model instance under test changes.
SOURCE_LON0 = 2.13 * u.deg
SOURCE_LAT0 = -0.37 * u.deg

ASYMPTOTIC_LON = 20.0 * u.deg
ASYMPTOTIC_LAT = 0.0 * u.deg
ASYMPTOTIC_LON0 = 2.0 * u.deg
ASYMPTOTIC_LAT0 = 0.0 * u.deg

NORM_LON = np.linspace(-5.0, 7.0, 601) * u.deg
NORM_LAT = np.linspace(-4.0, 4.0, 401) * u.deg

CENTER_LON = np.linspace(-1.0, 1.0, 201) * u.deg
CENTER_LAT = np.linspace(-1.0, 1.0, 201) * u.deg

SOURCE_LON = 2.0 * u.deg
SOURCE_LAT = 0.0 * u.deg

NORM_REL = 0.05
NORM_ABS = 0.02


@pytest.fixture(params=["coma_psf", "zernike_psf"])
def psf_model(request):
    return request.getfixturevalue(request.param)


def test_asymptotic_behavior(psf_model):
    assert np.isclose(
        psf_model.pdf(
            tel_id=1,
            lon=ASYMPTOTIC_LON,
            lat=ASYMPTOTIC_LAT,
            lon0=ASYMPTOTIC_LON0,
            lat0=ASYMPTOTIC_LAT0,
        ),
        0.0,
        atol=1e-7,
    )


def test_normalization(psf_model):
    lon0 = SOURCE_LON0
    lat0 = SOURCE_LAT0

    lon = NORM_LON
    lat = NORM_LAT
    lon_grid, lat_grid = np.meshgrid(lon, lat, indexing="xy")

    pdf = psf_model.pdf(
        tel_id=1,
        lon=lon_grid,
        lat=lat_grid,
        lon0=lon0,
        lat0=lat0,
    )

    assert np.isfinite(pdf).all()
    assert np.all(pdf >= 0.0)

    integral = trapz_func(
        trapz_func(pdf, lon.to_value(u.deg), axis=1), lat.to_value(u.deg)
    )

    assert integral == pytest.approx(1.0, rel=NORM_REL, abs=NORM_ABS)


def test_normalization_at_camera_center(psf_model):
    lon0 = 0.0 * u.deg
    lat0 = 0.0 * u.deg

    lon = CENTER_LON
    lat = CENTER_LAT
    lon_grid, lat_grid = np.meshgrid(lon, lat, indexing="xy")

    pdf = psf_model.pdf(
        tel_id=1,
        lon=lon_grid,
        lat=lat_grid,
        lon0=lon0,
        lat0=lat0,
    )

    integral = trapz_func(
        trapz_func(pdf, lon.to_value(u.deg), axis=1), lat.to_value(u.deg)
    )

    assert integral == pytest.approx(1.0, rel=NORM_REL, abs=NORM_ABS)


def test_finite_at_source_position(psf_model):
    value = psf_model.pdf(
        tel_id=1,
        lon=SOURCE_LON,
        lat=SOURCE_LAT,
        lon0=ASYMPTOTIC_LON0,
        lat0=ASYMPTOTIC_LAT0,
    )

    assert np.isfinite(value)
    assert value > 0.0


def test_finite_at_camera_center(psf_model):
    value = psf_model.pdf(
        tel_id=1,
        lon=0.0 * u.deg,
        lat=0.0 * u.deg,
        lon0=0.0 * u.deg,
        lat0=0.0 * u.deg,
    )

    assert np.isfinite(value)
    assert value > 0.0


def test_for_missing_config_parameters(example_subarray):
    with pytest.raises(
        ValueError, match="Missing ComaPSFModel configuration parameters:"
    ):
        PSFModel.from_name(
            "ComaPSFModel",
            subarray=example_subarray,
        )
