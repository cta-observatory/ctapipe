# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module is intended to contain astronomy-related helper tools which are
not provided by external packages and/or to satisfy particular needs of
usage within ctapipe.
"""

import logging
from pathlib import Path

import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.table import Table
from astropy.time import Time
from astroquery.vizier import Vizier

log = logging.getLogger("main")

__all__ = ["get_bright_stars_with_motion", "get_bright_stars"]


CACHE_FILE = Path("~/.psf_stars.ecsv").expanduser()


def cut_stars(stars, pointing=None, radius=None, Bmag_cut=None, Vmag_cut=None):
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
    if Vmag_cut is not None and "Vmag" in stars.keys():
        stars = stars[stars["Vmag"] < Vmag_cut]

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


def get_bright_stars_with_motion(
    pointing=None, radius=None, Bmag_cut=None, Vmag_cut=None, catalog="V/50/catalog"
):  # max_magnitude):
    """
    Get an astropy table of bright stars from a VizieR catalog

    You can browse the catalogs at https://vizier.cds.unistra.fr/viz-bin/VizieR

    Parameters
    ----------
    pointing: astropy Skycoord
       pointing direction in the sky (if none is given, full sky is returned)
    radius: astropy angular units
       Radius of the sky region around pointing position. Default: full sky
    Vmag_cut: float
        Return only stars above a given apparent magnitude. Default: None (all entries)
    Bmag_cut: float
        Return only stars above a given absolute magnitude. Default: None (all entries)
    catalog: string
        The location of the catalog to be used. The hipparcos catalog is found at I/239/hip_main. Default: V/50/catalog

    Returns
    -------
    Astropy table:
       List of all stars after cuts with names, catalog numbers, magnitudes,
       and coordinates
    """

    stars = None

    if CACHE_FILE.exists():
        log.info("Loading stars from cached table")
        try:
            stars = Table.read(CACHE_FILE)
            if Bmag_cut is not None:
                if stars.meta["Bmag_cut"] >= Bmag_cut:
                    log.debug(f"Loaded table is valid for { Bmag_cut= }")
                else:
                    log.debug("Loaded cache table has smaller magnitude_cut, reloading")
                    stars = None
            if Vmag_cut is not None:
                if stars.meta["Vmag_cut"] >= Vmag_cut:
                    log.debug(f"Loaded table is valid for {Vmag_cut= }")
                else:
                    log.debug("Loaded cache table has smaller magnitude_cut, reloading")
                    stars = None
        except Exception:
            log.exception("Cache file exists but reading failed. Recreating")

    if stars is None:
        log.info("Querying Vizier for bright stars catalog")
        # query vizier for stars with 0 <= Vmag <= max_magnitude
        vizier = Vizier(
            catalog=catalog,
            columns=["HR", "RAJ2000", "DEJ2000", "pmRA", "pmDE", "Vmag"],
            row_limit=1000000,
        )

        stars = vizier.query_constraints(Vmag="0.0..100.0")[0]
        if "Bmag" in stars.keys():
            if Bmag_cut is not None:
                stars.meta["Bmag_cut"] = Bmag_cut
        elif Bmag_cut is not None:
            log.exception("The chosen catalog does not have Bmag data")
        if "Vmag" in stars.keys():
            if Vmag_cut is not None:
                stars.meta["Vmag_cut"] = Vmag_cut
        elif Vmag_cut is not None:
            log.exception("The chosen catalog does not have Vmag data")

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
        stars, pointing=pointing, radius=radius, Bmag_cut=Bmag_cut, Vmag_cut=Vmag_cut
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
        catalog, pointing=pointing, radius=radius, Vmag_cut=magnitude_cut
    )

    catalog.remove_columns(["RAJ2000", "DEJ2000"])

    return catalog
