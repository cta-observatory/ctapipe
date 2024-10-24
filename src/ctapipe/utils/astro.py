# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module is intended to contain astronomy-related helper tools which are
not provided by external packages and/or to satisfy particular needs of
usage within ctapipe.
"""

import logging
from copy import deepcopy
from enum import Enum

import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time

log = logging.getLogger("main")

__all__ = ["get_star_catalog", "get_bright_stars"]


class StarCatalogues(Enum):
    Yale = {
        "directory": "V/50/catalog",
        "coordinates": {
            "frame": "icrs",
            "epoch": "J2000.0",
            "RA": {"column": "RAJ2000", "unit": "h:m:s"},
            "DE": {"column": "DEJ2000", "unit": "d:m:s"},
            "pmRA": {"column": "pmRA", "unit": "arsec/yr"},
            "pmDE": {"column": "pmDE", "unit": "arsec/yr"},
        },
        #: Vmag is mandatory (used for initial magnitude cut)
        "columns": ["RAJ2000", "DEJ2000", "pmRA", "pmDE", "Vmag", "HR"],
        "record": "yale_bright_star_catalog",
    }  #: Yale bright star catalogue
    Hipparcos = {
        "directory": "I/239/hip_main",
        "coordinates": {
            "frame": "icrs",
            "epoch": "J1991.25",
            "RA": {"column": "RAICRS", "unit": "deg"},
            "DE": {"column": "DEICRS", "unit": "deg"},
            "pmRA": {"column": "pmRA", "unit": "mas/yr"},
            "pmDE": {"column": "pmDE", "unit": "mas/yr"},
        },
        #: Vmag is mandatory (used for initial magnitude cut)
        "columns": ["RAICRS", "DEICRS", "pmRA", "pmDE", "Vmag", "BTmag", "HIP"],
        "record": "hipparcos_star_catalog",
    }  #: hipparcos catalogue


def select_stars(stars, pointing=None, radius=None, magnitude_cut=None, band="Vmag"):
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

    stars_ = deepcopy(stars)
    if magnitude_cut:
        try:
            stars_ = stars_[stars_[band] < magnitude_cut]
        except KeyError:
            raise ValueError(
                f"The requested catalogue has no compatible magnitude for the {band} band"
            )

    if radius:
        if pointing:
            stars_["separation"] = stars_["ra_dec"].separation(pointing)
            stars_ = stars_[stars_["separation"] < radius]
        else:
            raise ValueError(
                "Sky pointing, pointing=SkyCoord(), must be "
                "provided if radius is given."
            )

    return stars_


def get_star_catalog(catalog, magnitude_cutoff=8.0, row_limit=1000000):
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

    vizier = Vizier(
        catalog=catalog_dict["directory"],
        columns=catalog_dict["columns"],
        row_limit=row_limit,
    )

    stars = vizier.query_constraints(Vmag=f"<{magnitude_cutoff}")[0]

    stars.meta["Catalog"] = StarCatalogues[catalog].value

    return stars


def get_bright_stars(
    time, catalog="Yale", pointing=None, radius=None, magnitude_cut=None
):
    """
    Get an astropy table of bright stars from the Yale bright star catalog

    Parameters
    ----------
    time: astropy Time
        time to which proper motion is applied
    catalog : string
        name of the catalog to be used available catalogues are 'Yale' and 'Hipparcos'. Default: 'Yale'
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

    cat = StarCatalogues[catalog].value

    stars = get_table_dataset(cat["record"], role="bright star catalog")

    stars["ra_dec"] = SkyCoord(
        ra=Angle(
            stars[cat["coordinates"]["RA"]["column"]],
            unit=u.Unit(cat["coordinates"]["RA"]["unit"]),
        ),
        dec=Angle(
            stars[cat["coordinates"]["DE"]["column"]],
            unit=u.Unit(cat["coordinates"]["DE"]["unit"]),
        ),
        pm_ra_cosdec=stars[
            cat["coordinates"]["pmRA"]["column"]
            * u.Unit(cat["coordinates"]["pmRA"]["unit"])
        ],
        pm_dec=stars[
            cat["coordinates"]["pmDE"]["column"]
            * u.Unit(cat["coordinates"]["pmDE"]["unit"])
        ],
        frame=cat["coordinates"]["frame"],
        obstime=Time(cat["coordinates"]["epoch"]),
    )
    stars["ra_dec"] = stars["ra_dec"].apply_space_motion(new_obstime=time)
    stars["ra_dec"] = SkyCoord(
        stars["ra_dec"].ra, stars["ra_dec"].dec, obstime=stars["ra_dec"].obstime
    )

    stars.remove_columns(
        [cat["coordinates"]["RA"]["column"], cat["coordinates"]["DE"]["column"]]
    )

    stars = select_stars(
        stars, pointing=pointing, radius=radius, magnitude_cut=magnitude_cut
    )

    return stars
