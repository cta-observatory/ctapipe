# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
from astropy import units as u
import logging

logger = logging.getLogger(__name__)

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
       pointing direction in the sky
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
    from astropy.table import Table
    from ctapipe.utils import get_dataset_path

    catalog = get_dataset_path("yale_bright_star_catalog5.fits.gz")
    table = Table.read(catalog)

    starpositions = SkyCoord(ra=Angle(table['RAJ2000'], unit=u.deg),
                             dec=Angle(table['DEJ2000'], unit=u.deg),
                             frame='icrs', copy=False)
    table['ra_dec'] = starpositions

    if magnitude_cut is not None:
        table = table[table['Vmag'] < magnitude_cut]

    if radius is not None:
        separations = starpositions.separation(pointing)
        table['separation'] = separations
        table = table[separations < radius]

    table.remove_columns(['RAJ2000', 'DEJ2000'])

    return table

