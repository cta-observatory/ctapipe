# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module contains the utils.astro unit tests
"""
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord

from ..astro import get_bright_stars


def test_get_bright_stars():
    """
    unit test for utils.astro.get_bright_stars_with_motion().
    """
    # TODO add tests for all catalogues, specifically by trying to find some particular bright star, also test that motion is properly included
    pointing = SkyCoord(
        ra=Angle("03 47 29.1", unit=u.deg),
        dec=Angle("+24 06 18", unit=u.deg),
        frame="icrs",
    )

    # lets find 25 Eta Tau

    table = get_bright_stars(pointing=pointing, radius=1.0 * u.deg, Vmag_cut=3.5)

    assert len(table) == 1  # this looks if 25 Eta Tau was found
