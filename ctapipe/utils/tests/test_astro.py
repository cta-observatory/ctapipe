# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module contains the utils.astro unit tests
"""
from ..astro import get_bright_stars
from astropy.coordinates import SkyCoord
from astropy import units as u

def test_get_bright_stars():
    """
    unit test for utils.astro.get_bright_stars(). Tests that only Zeta Tau is
    returned close to the Crab Nebula as object brighter than mag=3.5.
    """
    pointing = SkyCoord(ra=83.275 * u.deg, dec=21.791 * u.deg, frame='icrs')

    table = get_bright_stars(pointing, radius=2. * u.deg, magnitude_cut=3.5)

    assert len(table) == 1
    assert table[0]['Name'] == '123Zet Tau'
