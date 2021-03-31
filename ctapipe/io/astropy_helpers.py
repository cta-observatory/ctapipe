#!/usr/bin/env python3
"""
Functions to help adapt internal ctapipe data to astropy formats and conventions
"""

from pathlib import Path

import tables
from astropy.table import Table
from astropy.units import Unit
from astropy.time import Time
import numpy as np

from .tableio import (
    FixedPointColumnTransform,
    QuantityColumnTransform,
    TimeColumnTransform,
)
from .hdf5tableio import get_hdf5_attr

__all__ = ["h5_table_to_astropy"]


def read_table(h5file, path) -> Table:
    """Get a table from a ctapipe-format HDF5 table as an `astropy.table.Table`
    object, retaining units. This uses the same unit storage convention as
    defined by the `HDF5TableWriter`, namely that the units are in attributes
    named by `<column name>_UNIT` that are parsible by `astropy.units`. Columns
    that were Enums will remain as integers.

    Parameters
    ----------
    h5file: Union[str, Path, tables.file.File]
        input filename or PyTables file handle
    path: str
        path to table in the file

    Returns
    -------
    astropy.table.Table:
        table in Astropy Format

    """

    should_close_file = False
    if isinstance(h5file, (str, Path)):
        h5file = tables.open_file(h5file)
        should_close_file = True
    elif isinstance(h5file, tables.file.File):
        pass
    else:
        raise ValueError(
            f"expected a string, Path, or PyTables "
            f"filehandle for argument 'h5file', got {h5file}"
        )

    table = h5file.get_node(path)

    other_attrs = {}
    column_descriptions = {}
    column_transforms = {}
    for attr in table.attrs._f_list():  # pylint: disable=W0212
        if attr.endswith("_UNIT"):
            colname = attr[:-5]
            column_transforms[colname] = QuantityColumnTransform(unit=table.attrs[attr])
        elif attr.endswith("_DESC"):
            colname = attr[:-5]
            column_descriptions[colname] = str(table.attrs[attr])
        elif attr.endswith("_TIME_SCALE"):
            colname, _, _ = attr.rpartition("_TIME_SCALE")
            scale = table.attrs[attr].lower()
            fmt = get_hdf5_attr(table.attrs, f"{colname}_TIME_FORMAT", "mjd").lower()
            column_transforms[colname] = TimeColumnTransform(scale=scale, format=fmt)
        elif attr.endswith("_TRANSFORM_SCALE"):
            colname, _, _ = attr.rpartition("_TRANSFORM_SCALE")
            column_transforms[colname] = FixedPointColumnTransform(
                scale=table.attrs[attr],
                offset=table.attrs[f"{colname}_TRANSFORM_OFFSET"],
                source_dtype=table.attrs[f"{colname}_TRANSFORM_DTYPE"],
                target_dtype=table.col(colname).dtype,
            )
        else:
            # need to convert to str() here so they are python strings, not
            # numpy strings
            value = table.attrs[attr]
            other_attrs[attr] = str(value) if isinstance(value, np.str_) else value

    astropy_table = Table(table[:], meta=other_attrs)

    for column, tr in column_transforms.items():
        astropy_table[column] = tr.inverse(astropy_table[column])

    for column, desc in column_descriptions.items():
        astropy_table[column].description = desc

    if should_close_file:
        h5file.close()

    return astropy_table
