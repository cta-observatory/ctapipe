# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module is intended to contain astronomy-related helper tools which are
not provided by external packages and/or to satisfy particular needs of
usage within ctapipe.
"""
from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
from astropy import units as u

__all__ = ['get_bright_stars']


def get_bright_stars(pointing=None, radius=None, magnitude_cut=None):
    """
    Returns an astropy table containing star positions above a given magnitude within
    a given radius around a position in the sky, using the Yale bright star catalog
    which needs to be present in the ctapipe-extra package. The included Yale bright
    star catalog contains all 9096 stars, excluding the Nova objects present in the
    original catalog from  Hoffleit & Jaschek (1991),
    http://adsabs.harvard.edu/abs/1991bsc..book.....H, and is complete down to
    magnitude ~6.5, while the faintest included star has mag=7.96.

    Parameters
    ----------
    pointing: astropy Skycoord
       pointing direction in the sky (if none is given, full sky is returned)
    radius: astropy angular units
       Radius of the sky region around pointing position. Default: full sky
    magnitude_cut: float
        Return only stars above a given magnitude. Default: None (all entries)

    Returns
    -------
    Astropy table:
       List of all stars after cuts with names, catalog numbers, magnitudes,
       and coordinates
    """
    from ctapipe.utils import get_table_dataset

    catalog = get_table_dataset("yale_bright_star_catalog5",
                                role="bright star catalog")

    starpositions = SkyCoord(ra=Angle(catalog['RAJ2000'], unit=u.deg),
                             dec=Angle(catalog['DEJ2000'], unit=u.deg),
                             frame='icrs', copy=False)
    catalog['ra_dec'] = starpositions

    if magnitude_cut is not None:
        catalog = catalog[catalog['Vmag'] < magnitude_cut]

    if radius is not None:
        if pointing is None:
            raise ValueError('Sky pointing, pointing=SkyCoord(), must be '
                             'provided if radius is given.')
        separations = catalog['ra_dec'].separation(pointing)
        catalog['separation'] = separations
        catalog = catalog[separations < radius]

    catalog.remove_columns(['RAJ2000', 'DEJ2000'])

    return catalog

