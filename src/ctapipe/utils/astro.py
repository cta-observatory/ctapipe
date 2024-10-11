# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module is intended to contain astronomy-related helper tools which are
not provided by external packages and/or to satisfy particular needs of
usage within ctapipe.
"""

import gzip
import logging
from io import TextIOWrapper
from pathlib import Path

import astropy.units as u
import requests
from astropy.coordinates import Angle, SkyCoord
from astropy.table import Table
from astropy.time import Time
from astroquery.vizier import Vizier

log = logging.getLogger("main")

__all__ = ["get_bright_stars_with_motion", "get_bright_stars"]


CACHE_FILE = Path("~/.psf_stars.ecsv").expanduser()


def cut_stars(stars, pointing=None, radius=None, magnitude_cut=None):
    if magnitude_cut is not None:
        stars = stars[stars["Vmag"] < magnitude_cut]

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


def read_ident_name_file(ident=6, colname="name"):
    log.info(f"Downloading common identification file {ident}")
    r = requests.get(
        f"https://cdsarc.cds.unistra.fr/ftp/I/239/version_cd/tables/ident{ident}.doc.gz",
        stream=True,
    )
    r.raise_for_status()

    with r, gzip.GzipFile(fileobj=r.raw, mode="r") as gz:
        table = {"HR": [], colname: []}

        for line in TextIOWrapper(gz):
            name, hip = line.split("|")
            table["HR"].append(int(hip.strip()))
            table[colname].append(name.strip())

    return Table(table)


def get_bright_stars_with_motion(
    pointing=None, radius=None, magnitude_cut=None
):  # max_magnitude):
    """
    Get an astropy table of bright stars from the Bright Star Catalogue

    This catalog is the 5th Revised Edition by Hoffleit et.al. 1991

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

    stars = None

    if CACHE_FILE.exists():
        log.info("Loading stars from cached table")
        log.info(CACHE_FILE)
        try:
            stars = Table.read(CACHE_FILE)
            if magnitude_cut is not None:
                if stars.meta["magnitude_cut"] >= magnitude_cut:
                    log.debug(f"Loaded table is valid for { magnitude_cut= }")
                else:
                    log.debug("Loaded cache table has smaller magnitude_cut, reloading")
                    stars = None
            else:
                stars = None
        except Exception:
            log.exception("Cache file exists but reading failed. Recreating")

    if stars is None:
        log.info("Querying Vizier for bright stars catalog")
        # query vizier for stars with 0 <= Vmag <= max_magnitude
        bright_star = "V/50/catalog"
        vizier = Vizier(
            catalog=bright_star,
            columns=["HR", "RAJ2000", "DEJ2000", "pmRA", "pmDE", "Vmag"],
            row_limit=1000000,
        )
        if magnitude_cut is not None:
            stars = vizier.query_constraints(Vmag=f"0.0..{magnitude_cut}")[0]

        else:
            stars = vizier.query_constraints(Vmag="0.0..100.0")[0]

        stars.meta["magnitude_cut"] = magnitude_cut
        stars.write(CACHE_FILE, overwrite=True)

    stars["ra_dec"] = SkyCoord(
        ra=Angle(stars["RAJ2000"], unit=u.deg),
        dec=Angle(stars["DEJ2000"], unit=u.deg),
        pm_ra_cosdec=stars["pmRA"].quantity,  # yes, pmRA is already pm_ra_cosdec
        pm_dec=stars["pmDE"].quantity,
        frame="icrs",
        obstime=Time("J2000.0"),
    )

    stars = cut_stars(
        stars, pointing=pointing, radius=radius, magnitude_cut=magnitude_cut
    )

    return stars


def get_bright_stars(pointing=None, radius=None, magnitude_cut=None):
    """
    Get an astropy table of bright stars.

    This function is using the Yale bright star catalog, available through ctapipe
    data downloads.

    The included Yale bright star catalog contains all 9096 stars, excluding the
    Nova objects present in the original catalog and is complete down to magnitude
    ~6.5, while the faintest included star has mag=7.96. :cite:p:`bright-star-catalog`

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

    catalog = get_table_dataset("yale_bright_star_catalog5", role="bright star catalog")

    starpositions = SkyCoord(
        ra=Angle(catalog["RAJ2000"], unit=u.deg),
        dec=Angle(catalog["DEJ2000"], unit=u.deg),
        frame="icrs",
        copy=False,
    )
    catalog["ra_dec"] = starpositions

    catalog = cut_stars(
        catalog, pointing=pointing, radius=radius, magnitude_cut=magnitude_cut
    )

    catalog.remove_columns(["RAJ2000", "DEJ2000"])

    return catalog
