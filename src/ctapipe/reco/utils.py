import astropy.units as u
import numpy as np


def _default_column_for_length(default, n_rows):
    """Create a length-matching column filled with `default` (unit-safe)."""
    if n_rows == 0:
        if isinstance(default, u.Quantity):
            return u.Quantity([], unit=default.unit)
        return np.full(0, default)

    return default


def add_defaults_and_meta(table, container, prefix=None, add_tel_prefix=False):
    """
    Fill column descriptions and default values into table for container

    Parameters
    ----------
    table : astropy.table.Table
        the table to be filled
    container : ctapipe.core.Container
        the container class to add columns and descriptions to the table
    prefix : str
        prefix for the column names
    add_tel_prefix : bool
        If True, add a ``tel_`` prefix to the column names to signal it's
        telescope-wise quantity
    """
    if prefix is None:
        prefix = container.default_prefix

    n_rows = len(table)
    col_prefix = f"{prefix}_tel_" if add_tel_prefix else f"{prefix}_"

    for name, field in container.fields.items():
        if add_tel_prefix and name == "telescopes":
            continue

        colname = f"{col_prefix}{name}"

        if colname not in table.colnames and field.default is not None:
            table[colname] = _default_column_for_length(field.default, n_rows)

        if colname in table.colnames:
            table[colname].description = field.description

        # add column name without prefix to column meta, needed for container reading
        table[colname].meta["NAME"] = name
