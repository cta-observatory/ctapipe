#!/usr/bin/env python3
"""
Functions to help adapt internal ctapipe data to astropy formats and conventions
"""

from pathlib import Path

import tables
from astropy.table import Table
import numpy as np

from .tableio import (
    FixedPointColumnTransform,
    QuantityColumnTransform,
    TimeColumnTransform,
)
from .hdf5tableio import get_hdf5_attr

from contextlib import ExitStack

__all__ = ["read_table"]


def read_table(h5file, path, start=None, stop=None, step=None) -> Table:
    """Read a table from an HDF5 file

    This reads a table written in the ctapipe format table as an `astropy.table.Table`
    object, inversing the column transformations units.

    This uses the same conventions as the `~ctapipe.io.HDF5TableWriter`,
    with the exception of Enums, that will remain as integers.

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

    with ExitStack() as stack:

        if isinstance(h5file, (str, Path)):
            h5file = stack.enter_context(tables.open_file(h5file))
        elif isinstance(h5file, tables.file.File):
            pass
        else:
            raise ValueError(
                f"expected a string, Path, or PyTables "
                f"filehandle for argument 'h5file', got {h5file}"
            )

        table = h5file.get_node(path)
        transforms, descriptions, meta = _parse_hdf5_attrs(table)
        astropy_table = Table(table[slice(start, stop, step)], meta=meta)
        for column, tr in transforms.items():
            astropy_table[column] = tr.inverse(astropy_table[column])

        for column, desc in descriptions.items():
            astropy_table[column].description = desc

        return astropy_table


def _parse_hdf5_attrs(table):
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

    return column_transforms, column_descriptions, other_attrs
