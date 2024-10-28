# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module contains the utils.astro unit tests
"""

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, Angle, EarthLocation, SkyCoord
from astropy.time import Time

from ..astro import get_bright_stars


def test_get_bright_stars():
    """
    unit test for utils.astro.get_bright_stars_with_motion().
    """
    # I will use polaris as a reference
    from astroquery.vizier import Vizier

    vizier = Vizier(
        catalog="Nomad",
        columns=["RAJ2000", "DEJ2000", "pmRA", "pmDE"],
        row_limit=10,
    )

    polaris = vizier.query_object("polaris", radius=1 * u.Unit("arcsec"))[0][0]

    t = Time("J2024")

    polaris = SkyCoord(
        ra=Angle(polaris["RAJ2000"], unit="deg"),
        dec=Angle(polaris["DEJ2000"], unit="deg"),
        pm_ra_cosdec=polaris["pmRA"] * u.Unit("mas/yr"),
        pm_dec=polaris["pmDE"] * u.Unit("mas/yr"),
        obstime=Time("J2000"),
    )

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
