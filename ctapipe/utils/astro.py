# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
from astropy import units as u
import logging

logger = logging.getLogger(__name__)

__all__ = ['get_bright_stars']


def get_bright_stars(pointing=SkyCoord(ra=0. * u.rad, dec=0. * u.rad, frame='icrs'),
                     radius=180. * u.deg, magnitude_cut=7.96):
    """
    Returns an astropy table containing star positions above a given magnitude within
    a given radius around a position in the sky, using the Yale bright star catalog
    which needs to be present in the ctapipe-extra package.

    Parameters
    ----------
    pointing: astropy Skycoord
       pointing direction in the sky
    radius: astropy angular units
       Radius of the sky region around pointing position. Default: full sky
    magnitude_cut: float
        Return only stars above a given magnitude. Default: 7.96 (all entries)

    Returns
    -------
    Astropy table:
       List of all stars after cuts with names, catalog numbers, magnitudes,
       and coordinates
    """
    from astropy.io import fits
    from astropy.table import Table
    from ctapipe.utils import get_dataset_path

    catalog = get_dataset_path("yale_brigh2t_star_catalog5.fits.gz")
    hdulist = fits.open(catalog)
    table = Table(hdulist[1].data)

    starpositions = SkyCoord(ra=Angle(table['RAJ2000'], unit=u.deg),
                             dec=Angle(table['DEJ2000'], unit=u.deg), frame='icrs')
    table['ra_dec'] = starpositions

    if radius < 180. * u.deg:
        separations = starpositions.separation(pointing)
        table['separation'] = separations
        table = table[separations < radius]
    if magnitude_cut < 7.96:
        table = table[table['Vmag'] < magnitude_cut]
    table.remove_columns(['RAJ2000', 'DEJ2000'])

    return table

