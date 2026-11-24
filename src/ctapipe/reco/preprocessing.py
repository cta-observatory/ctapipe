"""
Prepare event data for machine learning.

The resulting features can be passed to estimators such as scikit-learn models.

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

from ..containers import SubarrayEventContainer

LOG = logging.getLogger(__name__)


__all__ = [
    "check_valid_rows",
    "collect_features",
    "table_to_float",
    "table_to_X",
    "horizontal_to_telescope",
    "telescope_to_horizontal",
]


def table_to_float(table: Table, dtype=np.float32) -> np.ndarray:
    """Convert a table to a float32 array, replacing inf/-inf with float min/max

    Parameters
    ----------
    table : astropy.table.Table
        Table whose numeric columns become features in their original column order.
    dtype : numpy.dtype or type, optional
        Floating-point dtype of the output array.
    """
    X = structured_to_unstructured(table.as_array(), dtype=dtype)
    np.nan_to_num(X, nan=np.nan, copy=False)
    return X


def check_valid_rows(table: Table, warn=True, log=LOG) -> np.ndarray:
    """Check for nans, returning a mask of the rows not containing any nans

    Parameters
    ----------
    table : astropy.table.Table
        Numeric feature columns to check for NaN values.
    warn : bool, optional
        Log per-column counts when NaNs are found.
    log : logging.Logger, optional
        Logger for reporting invalid values.
    """
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
    Build a feature matrix from rows with valid feature values.

    Return a NumPy array for scikit-learn and the mask identifying the rows that were
    kept.

    Parameters
    ----------
    table : astropy.table.Table
        Event table from which to select model features.
    features : list of str
        Names of feature columns in the order expected by the model.
    log : logging.Logger, optional
        Logger for reporting rows with NaN features.
    """
    feature_table = table[features]
    valid = check_valid_rows(feature_table, log=log)
    X = table_to_float(feature_table[valid])
    return X, valid


def collect_features(
    event: SubarrayEventContainer, tel_id: int, subarray_table=None
) -> Table:
    """Loop over all containers with features.

    Parameters
    ----------
    event : SubarrayEventContainer
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

    if event.dl0 is not None:
        features.update(event.dl0.trigger.as_dict(recursive=False, flatten=True))

    tel_event = event.tel[tel_id]
    if tel_event.dl1 is None or tel_event.dl1.parameters is None:
        raise ValueError(
            f"Feature collection requires DL1 parameters for telescope {tel_id}"
        )
    features.update(
        tel_event.dl1.parameters.as_dict(
            add_prefix=True,
            recursive=True,
            flatten=True,
        )
    )

    for dl2 in (tel_event.dl2, event.dl2):
        if dl2 is not None:
            features.update(
                dl2.as_dict(
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
    """Transform coordinates from horizontal coordinates into TelescopeFrame

    Parameters
    ----------
    alt : astropy.units.Quantity
        Event altitude in the horizontal frame.
    az : astropy.units.Quantity
        Event azimuth in the horizontal frame.
    pointing_alt : astropy.units.Quantity
        Altitude of the telescope pointing.
    pointing_az : astropy.units.Quantity
        Azimuth of the telescope pointing.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingFrameAttributeWarning)

        horizontal_coord = AltAz(alt=alt, az=az)
        pointing = AltAz(alt=pointing_alt, az=pointing_az)
        tel_frame = TelescopeFrame(telescope_pointing=pointing)

        tel_coord = horizontal_coord.transform_to(tel_frame)

    return tel_coord.fov_lon.to(u.deg), tel_coord.fov_lat.to(u.deg)


@u.quantity_input(lon=u.deg, lat=u.deg, pointing_alt=u.deg, pointing_az=u.deg)
def telescope_to_horizontal(lon, lat, pointing_alt, pointing_az):
    """Transform coordinates from TelescopeFrame into horizontal coordinates

    Parameters
    ----------
    lon : astropy.units.Quantity
        Field-of-view longitude relative to the telescope pointing.
    lat : astropy.units.Quantity
        Field-of-view latitude relative to the telescope pointing.
    pointing_alt : astropy.units.Quantity
        Altitude of the telescope pointing.
    pointing_az : astropy.units.Quantity
        Azimuth of the telescope pointing.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingFrameAttributeWarning)

        pointing = AltAz(alt=pointing_alt, az=pointing_az)
        tel_coord = TelescopeFrame(
            fov_lon=lon, fov_lat=lat, telescope_pointing=pointing
        )
        horizontal_coord = tel_coord.transform_to(AltAz())

    return horizontal_coord.alt.to(u.deg), horizontal_coord.az.to(u.deg)
