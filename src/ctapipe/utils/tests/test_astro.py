# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module contains the utils.astro unit tests
"""
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord

from ..astro import get_bright_stars, get_bright_stars_with_motion


def test_get_bright_stars_with_motion():
    """
    unit test for utils.astro.get_bright_stars_with_motion().
    """
    pointing = SkyCoord(
        ra=Angle("03 47 29.1", unit=u.deg),
        dec=Angle("+24 06 18", unit=u.deg),
        frame="icrs",
    )

    # lets find 25 Eta Tau

    table = get_bright_stars_with_motion(
        pointing=pointing, radius=1.0 * u.deg, Vmag_cut=3.5
    )

    assert len(table) == 1


def test_get_bright_stars():
    """
    unit test for utils.astro.get_bright_stars(). Tests that only Zeta Tau is
    returned close to the Crab Nebula as object brighter than mag=3.5.
    """
    pointing = SkyCoord(ra=83.275 * u.deg, dec=21.791 * u.deg, frame="icrs")

    table = get_bright_stars(pointing, radius=2.0 * u.deg, magnitude_cut=3.5)

    assert len(table) == 1
    assert table[0]["Name"] == "123Zet Tau"
