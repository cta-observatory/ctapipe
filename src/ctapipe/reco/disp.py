"""
Common functions related to the disp reconstruction.
"""

import warnings
from typing import Annotated

import astropy.units as u
import numpy as np
from astropy.table import Table

from ..containers import CoordinateFrameType
from .preprocessing import horizontal_to_telescope

__all__ = [
    "compute_true_disp",
]


def get_tel_pointing(table):
    """
    Get telescope pointing from telescope events table.

    Prefers columns telescope_pointing_{altitude,azimuth} but will fallback
    to subarray_pointing_{lat,lon,frame} for backwards compatibility with older
    datasets.

    Parameters
    ----------
    table : Table
        table of telescope events.

    Returns
    -------
    alt : u.Quantity[angle]
        pointing altitude
    az : u.Quantity[angle]
        pointing azimuth
    """
    prefix = "telescope_pointing"
    tel_alt = f"{prefix}_altitude"
    tel_az = f"{prefix}_azimuth"

    if {tel_alt, tel_az}.issubset(table.colnames):
        return table[tel_alt].quantity, table[tel_az].quantity

    prefix = "subarray_pointing"
    if not np.all(table[f"{prefix}_frame"] == CoordinateFrameType.ALTAZ.value):
        raise ValueError(
            "Subarray pointing information for disp computation"
            " has to be provided in horizontal coordinates"
        )

    warnings.warn(
        "Input table does not contain telescope pointings, falling back to array pointing"
    )
    return table[f"{prefix}_lat"].quantity, table[f"{prefix}_lon"].quantity


def compute_true_disp(
    table: Table, project_disp: bool = True
) -> Annotated[u.Quantity, "angle"]:
    """
    Compute true disp parameter from a table of DL1 events.

    Parameters
    ----------
    table:
        DL1 telescope events table as created by `ctapipe.io.TableLoader.read_telescope_events`.
    project_disp:
        If true, the true source position of the gamma-ray is projected onto the reconstructed
        shower axis for computing the disp norm.
        Otherwise, the euclidean distance from the center of gravity to the true source position
        is used.

    Returns
    -------
    disp:
        Disp as a signed quantity
    """
    pointing_alt, pointing_az = get_tel_pointing(table)

    fov_lon, fov_lat = horizontal_to_telescope(
        alt=table["true_alt"],
        az=table["true_az"],
        pointing_alt=pointing_alt,
        pointing_az=pointing_az,
    )

    # numpy's trigonometric functions need radians
    psi = table["hillas_psi"].quantity.to_value(u.rad)
    cog_lon = table["hillas_fov_lon"].quantity
    cog_lat = table["hillas_fov_lat"].quantity

    delta_lon = fov_lon - cog_lon
    delta_lat = fov_lat - cog_lat

    true_disp = np.cos(psi) * delta_lon + np.sin(psi) * delta_lat
    true_sign = np.sign(true_disp)

    if project_disp:
        true_norm = np.abs(true_disp)
    else:
        true_norm = np.sqrt((fov_lon - cog_lon) ** 2 + (fov_lat - cog_lat) ** 2)

    return true_norm * true_sign
