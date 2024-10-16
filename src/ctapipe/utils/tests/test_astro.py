# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module contains the utils.astro unit tests
"""
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time

from ..astro import get_bright_stars


def test_get_bright_stars():
    """
    unit test for utils.astro.get_bright_stars_with_motion().
    """
    # TODO add tests for all catalogues, specifically by trying to find some particular bright star, also test that motion is properly included
    eta_tau = SkyCoord(
        ra=Angle("03 47 29.1", unit=u.deg),
        dec=Angle("+24 06 18", unit=u.deg),
        frame="icrs",
    )
    obstime = Time("2020-01-01")

    # lets find 25 Eta Tau

    table = get_bright_stars(pointing=eta_tau, radius=1.0 * u.deg, Vmag_cut=3.5)

    assert len(table) == 1  # this looks if 25 Eita Tau was found
    # check that the star moves
    assert table[0]["ra_dec"].apply_space_motion(obstime).ra != eta_tau.ra
    assert table[0]["ra_dec"].apply_space_motion(obstime).dec != eta_tau.dec

    # now test the other catalog. First get object 766 from the catalog

    HIP_star = SkyCoord(
        ra=Angle("00 08 23.2585712", unit=u.deg),
        dec=Angle("+29 05 25.555166", unit=u.deg),
        frame="icrs",
    )

    table = get_bright_stars(
        pointing=HIP_star, radius=1.0 * u.deg, Vmag_cut=3.5, catalog="Hippoarcos"
    )

    assert len(table) == 1
    # now check that stars move
    assert table[0]["ra_dec"].apply_space_motion(obstime).ra != eta_tau.ra
    assert table[0]["ra_dec"].apply_space_motion(obstime).dec != eta_tau.dec
