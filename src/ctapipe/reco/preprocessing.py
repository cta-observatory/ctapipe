"""
Utility functions for preprocessing data before handing them to machine
learning algorithms like sklearn models.

These functions are mainly taken from https://github.com/fact-project/aict-tools
and adapted to work with astropy Tables instead of pandas dataframes
"""

import logging
import warnings

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz
from astropy.table import QTable, Table
from numpy.lib.recfunctions import structured_to_unstructured

from ctapipe.coordinates import MissingFrameAttributeWarning, TelescopeFrame

from ..containers import ArrayEventContainer

LOG = logging.getLogger(__name__)


__all__ = [
    "check_valid_rows",
    "collect_features",
    "table_to_float",
    "table_to_X",
    "horizontal_to_telescope",
    "telescope_to_horizontal",
    "calculate_true_disp",
]


def table_to_float(table: Table, dtype=np.float32) -> np.ndarray:
    """Convert a table to a float32 array, replacing inf/-inf with float min/max"""
    X = structured_to_unstructured(table.as_array(), dtype=dtype)
    np.nan_to_num(X, nan=np.nan, copy=False)
    return X


def check_valid_rows(table: Table, warn=True, log=LOG) -> np.ndarray:
    """Check for nans, returning a mask of the rows not containing any nans"""

    nans = np.array([np.isnan(col) for col in table.columns.values()]).T
    valid = ~nans.any(axis=1)

    if warn:
        nan_counts = np.count_nonzero(nans, axis=0)
        if (nan_counts > 0).any():
            nan_counts_str = ", ".join(
                f"{k}: {v}" for k, v in zip(table.colnames, nan_counts) if v > 0
            )
            log.warning("Data contains not-predictable events.")
            log.warning("Number of nan-values in columns: %s", nan_counts_str)

    return valid


def table_to_X(table: Table, features: list[str], log=LOG):
    """
    Extract features as numpy ndarray to be given to sklearn from input table
    dropping all events for which one or more training features are nan
    """
    feature_table = table[features]
    valid = check_valid_rows(feature_table, log=log)
    X = table_to_float(feature_table[valid])
    return X, valid


def collect_features(
    event: ArrayEventContainer, tel_id: int, subarray_table=None
) -> Table:
    """Loop over all containers with features.

    Parameters
    ----------
    event : ArrayEventContainer
        The event container from which to collect the features
    tel_id : int
        The telscope id for which to collect the features
    subarray_table : Table
        The subarray as "to_table("joined")", to be added to the features.

    Returns
    -------
    Table
    """
    features = {}

    features.update(event.trigger.as_dict(recursive=False, flatten=True))

    features.update(
        event.dl1.tel[tel_id].parameters.as_dict(
            add_prefix=True,
            recursive=True,
            flatten=True,
        )
    )

    features.update(
        event.dl2.tel[tel_id].as_dict(
            add_prefix=True,
            recursive=True,
            flatten=True,
            add_key=False,  # prefix is already the map key for dl2 stuff
        )
    )

    features.update(
        event.dl2.stereo.as_dict(
            add_prefix=True,
            recursive=True,
            flatten=True,
            add_key=False,  # prefix is already the map key for dl2 stuff
        )
    )

    if subarray_table is not None:
        # to include units in features
        if not isinstance(subarray_table, QTable):
            subarray_table = QTable(subarray_table, copy=False)

        features.update(subarray_table.loc[tel_id])

    return Table({k: [v] for k, v in features.items()})


@u.quantity_input(alt=u.deg, az=u.deg, pointing_alt=u.deg, pointing_az=u.deg)
def horizontal_to_telescope(alt, az, pointing_alt, pointing_az):
    """Transform coordinates from horizontal coordinates into TelescopeFrame"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingFrameAttributeWarning)

        horizontal_coord = AltAz(alt=alt, az=az)
        pointing = AltAz(alt=pointing_alt, az=pointing_az)
        tel_frame = TelescopeFrame(telescope_pointing=pointing)

        tel_coord = horizontal_coord.transform_to(tel_frame)

    return tel_coord.fov_lon.to(u.deg), tel_coord.fov_lat.to(u.deg)


@u.quantity_input(lon=u.deg, lat=u.deg, pointing_alt=u.deg, pointing_az=u.deg)
def telescope_to_horizontal(lon, lat, pointing_alt, pointing_az):
    """Transform coordinates from TelescopeFrame into horizontal coordinates"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingFrameAttributeWarning)

        pointing = AltAz(alt=pointing_alt, az=pointing_az)
        tel_coord = TelescopeFrame(
            fov_lon=lon, fov_lat=lat, telescope_pointing=pointing
        )
        horizontal_coord = tel_coord.transform_to(AltAz())

    return horizontal_coord.alt.to(u.deg), horizontal_coord.az.to(u.deg)


@u.quantity_input(
    fov_lon=u.deg, fov_lat=u.deg, hillas_psi=u.rad, hillas_lon=u.deg, hillas_lat=u.deg
)
def calculate_true_disp(fov_lon, fov_lat, hillas_psi, hillas_lon, hillas_lat):
    """
    Calculate the true disp parameter (distance between hillas cog and source position)

    Parameters
    ----------
    fov_lon : u.Quantity
        Source longitude in telescope frame
    fov_lat : u.Quantity
        Source latitude in telescope frame
    hillas_psi : u.Quantity
        Hillas psi parameter (angle between major axis and x-axis)
    hillas_lon : u.Quantity
        Hillas center longitude
    hillas_lat : u.Quantity
        Hillas center latitude

    Returns
    -------
    true_disp : u.Quantity
        The true disp parameter (signed distance)
    """
    delta_lon = fov_lon - hillas_lon
    delta_lat = fov_lat - hillas_lat

    true_disp = np.cos(hillas_psi) * delta_lon + np.sin(hillas_psi) * delta_lat
    return true_disp
