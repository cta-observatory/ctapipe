"""
Common functions related to the disp reconstruction.
"""

from typing import Annotated

import astropy.units as u
import numpy as np
from astropy.table import Table

from .preprocessing import horizontal_to_telescope

__all__ = [
    "compute_true_disp",
]


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
    fov_lon, fov_lat = horizontal_to_telescope(
        alt=table["true_alt"],
        az=table["true_az"],
        pointing_alt=table["subarray_pointing_lat"],
        pointing_az=table["subarray_pointing_lon"],
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
