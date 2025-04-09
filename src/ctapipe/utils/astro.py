# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module is intended to contain astronomy-related helper tools which are
not provided by external packages and/or to satisfy particular needs of
usage within ctapipe.
"""

import logging
from collections import namedtuple
from copy import deepcopy
from enum import Enum

import astropy.units as u
from astropy.coordinates import Angle, SkyCoord, UnitSphericalCosLatDifferential
from astropy.table import Table
from astropy.time import Time
from astropy.units import Quantity

log = logging.getLogger("main")

__all__ = ["get_star_catalog", "get_bright_stars", "StarCatalog", "CatalogInfo"]

# Define a namedtuple to hold the catalog information
CatalogInfo = namedtuple(
    "CatalogInfo", ["directory", "coordinates", "columns", "record"]
)


class StarCatalog(Enum):
    """
    Enumeration of star catalogs with their respective metadata.

    Each catalog entry is represented as a namedtuple `CatalogInfo` containing:

    Attributes
    ----------
    directory : str
        The directory path of the catalog in the Vizier database.
    coordinates : dict
        A dictionary containing the coordinate frame, epoch, and column names for RA and DE.
    columns : list of str
        A list of columns to be retrieved from the catalog.
    record : str
        A name of the catalog file in the cache.
    """

    Yale = CatalogInfo(
        directory="V/50/catalog",
        coordinates={
            "frame": "icrs",
            "epoch": "J2000.0",
            "RA": {"column": "RAJ2000", "unit": "hourangle"},
            "DE": {"column": "DEJ2000", "unit": "deg"},
        },
        #: Vmag is mandatory (used for initial magnitude cut)
        columns=["RAJ2000", "DEJ2000", "pmRA", "pmDE", "Vmag", "HR"],
        record="yale_bright_star_catalog",
    )  #: Yale bright star catalogue
    Hipparcos = CatalogInfo(
        directory="I/239/hip_main",
        coordinates={
            "frame": "icrs",
            "epoch": "J1991.25",
            "RA": {"column": "RAICRS", "unit": "deg"},
            "DE": {"column": "DEICRS", "unit": "deg"},
        },
        #: Vmag is mandatory (used for initial magnitude cut)
        columns=["RAICRS", "DEICRS", "pmRA", "pmDE", "Vmag", "BTmag", "HIP"],
        record="hipparcos_star_catalog",
    )  #: HIPPARCOS catalogue


def select_stars(
    stars: Table,
    pointing: SkyCoord = None,
    radius: Quantity = None,
    magnitude_cut: float = None,
    band: str = "Vmag",
) -> Table:
    """
    Utility function to filter stars based on magnitude and/or location.

    Parameters
    ----------
    stars : astropy.table.Table
        Table of stars, including magnitude and coordinates.
    pointing : astropy.coordinates.SkyCoord, optional
        Pointing direction in the sky. If None is given, the full sky is returned.
    radius : astropy.units.Quantity, optional
        Radius of the sky region around the pointing position. Default is the full sky.
    magnitude_cut : float, optional
        Return only stars above a given apparent magnitude. Default is None (all entries).
    band : str, optional
        Wavelength band to use for the magnitude cut. Options are 'Vmag'
        or any other optical band column name, present in the catalog (e.g. Bmag or BTmag, etc.).
        Default is 'Vmag'.

    Returns
    -------
    astropy.table.Table
        List of all stars after applying the cuts, with the same keys as the input table `stars`.
    """
    stars_ = deepcopy(stars)
    if magnitude_cut:
        try:
            stars_ = stars_[stars_[band] < magnitude_cut]
        except KeyError:
            raise ValueError(
                f"The requested catalogue has no compatible magnitude for the {band} band"
            )

    if radius is not None:
        if pointing:
            stars_["separation"] = stars_["ra_dec"].separation(pointing)
            stars_ = stars_[stars_["separation"] < radius]
        else:
            raise ValueError(
                "Sky pointing, pointing=SkyCoord(), must be "
                "provided if radius is given."
            )

    return stars_


def get_star_catalog(
    catalog: str | StarCatalog, magnitude_cutoff: float = 8.0, row_limit: int = 1000000
) -> Table:
    """
    Utility function to download a star catalog for the get_bright_stars function.

    Parameters
    ----------
    catalog : str or ctapipe.utils.astro.StarCatalog
        Name of the catalog to be used. Usable names are found in the Enum StarCatalog. Default: Yale.
    magnitude_cutoff : float, optional
        Maximum value for magnitude used in lookup. Default is 8.0.
    row_limit : int, optional
        Maximum number of rows for the star catalog lookup. Default is 1000000.

    Returns
    -------
    astropy.table.Table
        List of all stars after cuts with catalog numbers, magnitudes,
        and coordinates as SkyCoord objects including proper motion.
    """
    from astroquery.vizier import Vizier

    if isinstance(catalog, str):
        catalog = StarCatalog[catalog]
    catalog_info = catalog.value
    catalog_name = catalog.name

    vizier = Vizier(
        catalog=catalog_info.directory,
        columns=catalog_info.columns,
        row_limit=row_limit,
    )

    stars = vizier.query_constraints(Vmag=f"<{magnitude_cutoff}")[0]

    header = {
        "ORIGIN": "CTAPIPE",
        "JEPOCH": float(catalog_info.coordinates["epoch"].replace("J", "")),
        "RADESYS": catalog_info.coordinates["frame"].upper(),
        "MAGCUT": magnitude_cutoff,
        "BAND": "V",
        "CATALOG": catalog_name,
        "REFERENC": catalog_info.directory,
        "COLUMNS": "_".join(catalog_info.columns),
    }

    stars.meta = header

    return stars


def get_bright_stars(
    time: Time,
    catalog: StarCatalog | str = "Yale",
    pointing: SkyCoord | None = None,
    radius: Quantity | None = None,
    magnitude_cut: float | None = None,
    band: str = "Vmag",
) -> Table:
    """
    Get an astropy table of bright stars from the specified star catalog.

    Parameters
    ----------
    time : astropy.time.Time
        Time to which proper motion is applied.
    catalog : str or ctapipe.utils.astro.StarCatalog, optional
        Name of the catalog to be used. Available catalogues are 'Yale' and 'Hipparcos'. Default is 'Yale'.
    pointing : astropy.coordinates.SkyCoord, optional
        Pointing direction in the sky. If None is given, the full sky is returned.
    radius : astropy.units.Quantity, optional
        Radius of the sky region around the pointing position. Default is the full sky.
    magnitude_cut : float, optional
        Return only stars above a given absolute magnitude. Default is None (all entries).
    band : str, optional
        Wavelength band to use for the magnitude cut. Options are 'Vmag'
        or any other optical band column name, present in the catalog (e.g. Bmag or BTmag, etc.).
        Default is 'Vmag'.

    Returns
    -------
    astropy.table.Table
        List of all stars after applying the cuts, with catalog numbers, magnitudes,
        and coordinates as SkyCoord objects including proper motion.
    """
    from importlib.resources import as_file, files

    if isinstance(catalog, str):
        catalog = StarCatalog[catalog]
    cat = catalog.value
    record = cat.record

    with as_file(files("ctapipe").joinpath(f"resources/{record}.fits.gz")) as f:
        stars = Table.read(f)

    stars["ra_dec"] = SkyCoord(
        ra=Angle(
            stars[cat.coordinates["RA"]["column"]],
            unit=u.Unit(cat.coordinates["RA"]["unit"]),
        ),
        dec=Angle(
            stars[cat.coordinates["DE"]["column"]],
            unit=u.Unit(cat.coordinates["DE"]["unit"]),
        ),
        pm_ra_cosdec=stars["pmRA"].quantity,
        pm_dec=stars["pmDE"].quantity,
        frame=cat.coordinates["frame"],
        obstime=Time(cat.coordinates["epoch"]),
    )
    stars["ra_dec"] = stars["ra_dec"].apply_space_motion(new_obstime=time)
    stars["ra_dec"].data.differentials["s"] = (
        stars["ra_dec"]
        .data.differentials["s"]
        .represent_as(UnitSphericalCosLatDifferential)
    )
    stars.remove_columns(
        [cat.coordinates["RA"]["column"], cat.coordinates["DE"]["column"]]
    )

    stars = select_stars(
        stars, pointing=pointing, radius=radius, magnitude_cut=magnitude_cut, band=band
    )

    return stars
