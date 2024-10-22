# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module is intended to contain astronomy-related helper tools which are
not provided by external packages and/or to satisfy particular needs of
usage within ctapipe.
"""

import logging
from enum import Enum

import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time

log = logging.getLogger("main")

__all__ = ["get_bright_stars", "get_hipparcos_stars"]


class StarCatalogues(Enum):
    Yale = "V/50/catalog"  #: Yale bright star catalogue
    Hippoarcos = "I/239/hip_main"  #: hipparcos catalogue


def select_stars(stars, pointing=None, radius=None, Bmag_cut=None, Vmag_cut=None):
    """
    utility function to cut stars based on magnitude and/or location

    Parameters
    ----------
    stars: astropy table
        Table of stars, including magnitude and coordinates
    pointing: astropy Skycoord
        pointing direction in the sky (if none is given, full sky is returned)
    radius: astropy angular units
        Radius of the sky region around pointing position. Default: full sky
    Vmag_cut: float
        Return only stars above a given apparent magnitude. Default: None (all entries)
    Bmag_cut: float
        Return only stars above a given absolute magnitude. Default: None (all entries)

    Returns
    -------
    Astropy table:
       List of all stars after cuts with same keys as the input table stars
    """

    if Bmag_cut is not None and "Bmag" in stars.keys():
        stars = stars[stars["Bmag"] < Bmag_cut]
    elif Bmag_cut is not None and "BTmag" in stars.keys():
        stars = stars[stars["BTmag"] < Bmag_cut]
    if Vmag_cut is not None and "Vmag" in stars.keys():
        stars = stars[stars["Vmag"] < Vmag_cut]
    elif Vmag_cut is not None and "VTmag" in stars.keys():
        stars = stars[stars["VTmag"] < Vmag_cut]

    if radius is not None:
        if pointing is None:
            raise ValueError(
                "Sky pointing, pointing=SkyCoord(), must be "
                "provided if radius is given."
            )
        separations = stars["ra_dec"].separation(pointing)
        stars["separation"] = separations
        stars = stars[separations < radius]

    return stars


def get_star_catalog(catalog):
    """
    Utility function to download a star catalog for the get_bright_stars function

    Parameters
    ----------
    catalog: string
        Name of the catalog to be used. Usable names are found in the Enum StarCatalogues. Default: Yale

    Returns
    ----------
    Astropy table:
       List of all stars after cuts with catalog numbers, magnitudes,
       and coordinates as SkyCoord objects including proper motion
    """
    from astroquery.vizier import Vizier

    vizier = Vizier(
        catalog=StarCatalogues[catalog].value,
        columns=[
            "HIP",  #: HIP is the Hippoarcos ID available for that catalog
            "HR",  #: HR is the Harvard Revised Number available for the Yale catalog
            "RAJ2000",
            "DEJ2000",
            "RAICRS",
            "DEICRS",
            "pmRA",
            "pmDE",
            "Vmag",
            "Bmag",
        ],
        row_limit=1000000,
    )

    stars = vizier.query_constraints(Vmag="0.0..10.0")[0]

    stars.meta["Catalog"] = StarCatalogues[catalog].value

    return stars


def get_bright_stars(pointing=None, radius=None, magnitude_cut=None):
    """
    Get an astropy table of bright stars from the Yale bright star catalog

    Parameters
    ----------
    pointing: astropy Skycoord
       pointing direction in the sky (if none is given, full sky is returned)
    radius: astropy angular units
       Radius of the sky region around pointing position. Default: full sky
    magnitude_cut: float
        Return only stars above a given absolute magnitude. Default: None (all entries)

    Returns
    -------
    Astropy table:
       List of all stars after cuts with catalog numbers, magnitudes,
       and coordinates as SkyCoord objects including proper motion
    """

    from ctapipe.utils import get_table_dataset

    stars = get_table_dataset("yale_bright_star_catalog5", role="bright star catalog")

    stars["ra_dec"] = SkyCoord(
        ra=Angle(stars["RAJ2000"], unit=u.deg),
        dec=Angle(stars["DEJ2000"], unit=u.deg),
        pm_ra_cosdec=stars["pmRA"].quantity,  # yes, pmRA is already pm_ra_cosdec
        pm_dec=stars["pmDE"].quantity,
        frame="icrs",
        obstime=Time("J2000.0"),
    )
    stars.remove_columns(["RAJ2000", "DEJ2000"])

    stars = select_stars(
        stars, pointing=pointing, radius=radius, Vmag_cut=magnitude_cut
    )

    return stars


def get_hipparcos_stars(pointing=None, radius=None, magnitude_cut=None):
    """
    Get an astropy table of bright stars from the Hippoarcos star catalog

    Parameters
    ----------
    pointing: astropy Skycoord
       pointing direction in the sky (if none is given, full sky is returned)
    radius: astropy angular units
       Radius of the sky region around pointing position. Default: full sky
    magnitude_cut: float
        Return only stars above a given absolute magnitude. Default: None (all entries)

    Returns
    -------
    Astropy table:
       List of all stars after cuts with catalog numbers, magnitudes,
       and coordinates as SkyCoord objects including proper motion
    """

    from ctapipe.utils import get_table_dataset

    stars = get_table_dataset("hippoarcos_star_catalog5", role="bright star catalog")

    stars["ra_dec"] = SkyCoord(
        ra=Angle(stars["RAICRS"], unit=u.deg),
        dec=Angle(stars["DEICRS"], unit=u.deg),
        pm_ra_cosdec=stars["pmRA"].quantity,  # yes, pmRA is already pm_ra_cosdec
        pm_dec=stars["pmDE"].quantity,
        frame="icrs",
        obstime=Time("J1991.25"),
    )
    stars.remove_columns(["RAICRS", "DEICRS"])

    stars = select_stars(
        stars, pointing=pointing, radius=radius, Vmag_cut=magnitude_cut
    )

    return stars
