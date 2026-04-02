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


def test_asymptotic_behavior(coma_psf):
    assert np.isclose(
        coma_psf.pdf(
            tel_id=1,
            lon=20.0 * u.deg,
            lat=0.0 * u.deg,
            lon0=2.0 * u.deg,
            lat0=0.0 * u.deg,
        ),
        0.0,
        atol=1e-7,
    )


def test_normalization(coma_psf):
    lon0 = 2.13 * u.deg
    lat0 = -0.37 * u.deg

    lon = np.linspace(-5.0, 7.0, 601) * u.deg
    lat = np.linspace(-4.0, 4.0, 401) * u.deg
    lon_grid, lat_grid = np.meshgrid(lon, lat, indexing="xy")

    pdf = coma_psf.pdf(
        tel_id=1,
        lon=lon_grid,
        lat=lat_grid,
        lon0=lon0,
        lat0=lat0,
    )

    integral = trapz_func(
        trapz_func(pdf, lon.to_value(u.deg), axis=1), lat.to_value(u.deg)
    )

    assert integral == pytest.approx(1.0, rel=0.05, abs=0.02)


def test_normalization_at_camera_center(coma_psf):
    lon0 = 0.0 * u.deg
    lat0 = 0.0 * u.deg

    lon = np.linspace(-1.0, 1.0, 201) * u.deg
    lat = np.linspace(-1.0, 1.0, 201) * u.deg
    lon_grid, lat_grid = np.meshgrid(lon, lat, indexing="xy")

    pdf = coma_psf.pdf(
        tel_id=1,
        lon=lon_grid,
        lat=lat_grid,
        lon0=lon0,
        lat0=lat0,
    )

    integral = trapz_func(
        trapz_func(pdf, lon.to_value(u.deg), axis=1), lat.to_value(u.deg)
    )

    assert integral == pytest.approx(1.0, rel=0.05, abs=0.02)


def test_finite_at_source_position(coma_psf):
    value = coma_psf.pdf(
        tel_id=1,
        lon=2.0 * u.deg,
        lat=0.0 * u.deg,
        lon0=2.0 * u.deg,
        lat0=0.0 * u.deg,
    )

    assert np.isfinite(value)
    assert value > 0.0


def test_finite_at_camera_center(coma_psf):
    value = coma_psf.pdf(
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
