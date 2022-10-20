"""
Utility functions for preprocessing data before handing them to machine
learning algorithms like sklearn models.

These functions are mainly taken from https://github.com/fact-project/aict-tools
and adapted to work with astropy Tables instead of pandas dataframes
"""
import logging

import numpy as np
from astropy.table import QTable, Table
from numpy.lib.recfunctions import structured_to_unstructured

from ..containers import ArrayEventContainer

LOG = logging.getLogger(__name__)


__all__ = ["table_to_float", "check_valid_rows"]


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
