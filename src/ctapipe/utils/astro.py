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
    Yale = {
        "directory": "V/50/catalog",
        "band": ["V"],
        "frame": "J2000",
        "ID_type": "HR",
    }  #: Yale bright star catalogue
    Hipparcos = {
        "directory": "I/239/hip_main",
        "band": ["V", "B"],
        "frame": "ICRS",
        "ID_type": "HIP",
    }  #: hipparcos catalogue


def select_stars(stars, pointing=None, radius=None, magnitude_cut=None, band="B"):
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
    magnitude_cut: float
        Return only stars above a given apparent magnitude. Default: None (all entries)
    band: string
        wavelength band to use for the magnitude cut options are V and B. Default: 'B' (all entries)

    Returns
    -------
    Astropy table:
       List of all stars after cuts with same keys as the input table stars
    """

    if magnitude_cut is not None:
        if band == "B":
            if "Bmag" in stars.keys():
                stars = stars[stars["Bmag"] < magnitude_cut]
            elif "BTmag" in stars.keys():
                stars = stars[stars["BTmag"] < magnitude_cut]
            else:
                raise ValueError(
                    "The requested catalogue has no compatible magnitude for the B band"
                )

        if band == "V":
            if "Vmag" in stars.keys():
                stars = stars[stars["Vmag"] < magnitude_cut]
            elif "VTmag" in stars.keys():
                stars = stars[stars["VTmag"] < magnitude_cut]
            else:
                raise ValueError(
                    "The requested catalogue has no compatible magnitude for the V band"
                )

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


def get_star_catalog(catalog, min_magnitude=0.0, max_magnitude=10.0, row_limit=1000000):
    """
    Utility function to download a star catalog for the get_bright_stars function

    Parameters
    ----------
    catalog: string
        Name of the catalog to be used. Usable names are found in the Enum StarCatalogues. Default: Yale
    min_magnitude: float
        Minimum value for magnitude used in lookup
    max_magnitude: float
        Maximum value for magnitude used in lookup
    row_limit: int
        Maximum number of rows for the star catalog lookup

    Returns
    ----------
    Astropy table:
       List of all stars after cuts with catalog numbers, magnitudes,
       and coordinates as SkyCoord objects including proper motion
    """
    from astroquery.vizier import Vizier

    catalog_dict = StarCatalogues[catalog].value

    columns = ["pmRA", "pmDE", catalog_dict["ID_type"]]
    if "B" in catalog_dict["band"]:
        columns.append("Bmag")
        columns.append("BTmag")
    elif "V" in catalog_dict["band"]:
        columns.append("Vmag")
        columns.append("VTmag")
    if catalog_dict["frame"] == "J2000":
        columns.append("RAJ2000")
        columns.append("DEJ2000")
    elif catalog_dict["frame"] == "ICRS":
        columns.append("RAICRS")
        columns.append("DEICRS")

    vizier = Vizier(
        catalog=catalog_dict["directory"],
        columns=columns,
        row_limit=row_limit,
    )

    stars = vizier.query_constraints(Vmag="{min_magnitude}..{max_magnitude}")[0]

    stars.meta["Catalog"] = StarCatalogues[catalog].value

    return stars


def get_bright_stars(time, pointing=None, radius=None, magnitude_cut=None):
    """
    Get an astropy table of bright stars from the Yale bright star catalog

    Parameters
    ----------
    time: astropy Time
        time to which proper motion is applied
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
        frame="galactic",
        obstime=Time("J2000.0"),
    )
    stars["ra_dec"].apply_space_motion(new_obstime=time)
    stars.remove_columns(["RAJ2000", "DEJ2000"])

    stars = select_stars(
        stars, pointing=pointing, radius=radius, Vmag_cut=magnitude_cut
    )

    return stars


def get_hipparcos_stars(time, pointing=None, radius=None, magnitude_cut=None):
    """
    Get an astropy table of bright stars from the Hippoarcos star catalog

    Parameters
    ----------
    time: astropy Time
        time to which proper motion is applied
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

    stars = get_table_dataset("hipparcos_star_catalog5", role="bright star catalog")

    stars["ra_dec"] = SkyCoord(
        ra=Angle(stars["RAICRS"], unit=u.deg),
        dec=Angle(stars["DEICRS"], unit=u.deg),
        pm_ra_cosdec=stars["pmRA"].quantity,  # yes, pmRA is already pm_ra_cosdec
        pm_dec=stars["pmDE"].quantity,
        frame="icrs",
        obstime=Time("J1991.25"),
    )
    stars["ra_dec"].apply_space_motion(new_obstime=time)
    stars.remove_columns(["RAICRS", "DEICRS"])

    stars = select_stars(
        stars, pointing=pointing, radius=radius, Vmag_cut=magnitude_cut
    )

    return stars
