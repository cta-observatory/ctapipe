# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module contains the utils.astro unit tests
"""

import warnings

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.table import Table
from astropy.time import Time
from erfa import ErfaWarning


@pytest.mark.vizier
def test_get_bright_stars():
    """
    unit test for utils.astro.get_bright_stars_with_motion().
    """
    from ctapipe.utils import get_bright_stars

    t = Time("J2024")

    polaris = SkyCoord(
        ra=37.9545108 * u.deg,
        dec=89.2641097 * u.deg,
        pm_ra_cosdec=44.2 * u.mas / u.year,
        pm_dec=-11.7 * u.mas / u.year,
        obstime=Time("J2000"),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ErfaWarning)
        polaris_2024 = polaris.apply_space_motion(t)

    table_yale = get_bright_stars(t, pointing=polaris_2024, radius=1 * u.Unit("arcsec"))
    table_hip = get_bright_stars(
        t, catalog="Hipparcos", pointing=polaris_2024, radius=1.0 * u.Unit("arcsec")
    )

    assert len(table_yale) == 1  # this looks if
    assert len(table_hip) == 1
    assert np.isclose(
        table_yale[0]["ra_dec"].ra.to_value(unit="deg"),
        polaris_2024.ra.to_value(unit="deg"),
        rtol=0.01,
    )
    assert np.isclose(
        table_yale[0]["ra_dec"].dec.to_value(unit="deg"),
        polaris_2024.dec.to_value(unit="deg"),
        rtol=0.01,
    )
    assert np.isclose(
        table_hip[0]["ra_dec"].ra.to_value(unit="deg"),
        polaris_2024.ra.to_value(unit="deg"),
        rtol=0.01,
    )
    assert np.isclose(
        table_hip[0]["ra_dec"].dec.to_value(unit="deg"),
        polaris_2024.dec.to_value(unit="deg"),
        rtol=0.01,
    )
    #  Check that the coordinate transformation works
    location = EarthLocation(
        lat=28.7616 * u.deg, lon=-17.8914 * u.deg, height=2200 * u.m
    )
    table_yale["ra_dec"].transform_to(AltAz(location=location, obstime=t))


@pytest.mark.vizier
@pytest.mark.filterwarnings(
    "error::astropy.coordinates.errors.NonRotationTransformationWarning"
)
@pytest.mark.filterwarnings("error::erfa.core.ErfaWarning")
def test_warning():
    """Test that get_bright_stars with radius given does not issue the NonRotationTransformationWarning."""
    from ctapipe.utils import get_bright_stars

    location = EarthLocation.of_site("Roque de los Muchachos")

    obstime = Time("2025-01-01T23:00", scale="utc")
    az_tel = 180 * u.deg
    alt_tel = 70 * u.deg

    horizon_frame = AltAz(location=location, obstime=obstime)

    pointing = SkyCoord(az=az_tel, alt=alt_tel, frame=horizon_frame)

    get_bright_stars(time=obstime, pointing=pointing, radius=3 * u.deg, magnitude_cut=8)


@pytest.mark.vizier
def test_update_catalogs(tmp_path):
    from ctapipe.utils.astro import update_star_catalogs

    update_star_catalogs(tmp_path)

    hipparcos = Table.read(tmp_path / "hipparcos_star_catalog.fits.gz")
    assert "name" in hipparcos.colnames
    assert "flamsteed" in hipparcos.colnames

    yale = Table.read(tmp_path / "yale_bright_star_catalog.fits.gz")
    assert "Name" in yale.colnames
