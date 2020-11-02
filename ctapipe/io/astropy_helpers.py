#!/usr/bin/env python3
"""
Functions to help adapt internal ctapipe data to astropy formats and conventions
"""

from pathlib import Path

import tables
from astropy.table import QTable
from astropy.units import Unit
import numpy as np

__all__ = ["h5_table_to_astropy"]


def h5_table_to_astropy(h5file, path) -> QTable:
    """Get a table from a ctapipe-format HDF5 table as an `astropy.table.QTable`
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
    astropy.table.QTable:
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
    column_units = {}  # mapping of colname to unit
    column_descriptions = {}
    for attr in table.attrs._f_list():  # pylint: disable=W0212
        if attr.endswith("_UNIT"):
            colname = attr[:-5]
            column_units[colname] = table.attrs[attr]
        elif attr.endswith("_DESC"):
            colname = attr[:-5]
            column_descriptions[colname] = str(table.attrs[attr])
        else:
            # need to convert to str() here so they are python strings, not
            # numpy strings
            value = table.attrs[attr]
            other_attrs[attr] = str(value) if isinstance(value, np.str_) else value

    astropy_table = QTable(table[:], meta=other_attrs)

    for column, unit in column_units.items():
        astropy_table[column].unit = Unit(unit)

    for column, desc in column_descriptions.items():
        astropy_table[column].description = desc

    if should_close_file:
        h5file.close()

    return astropy_table
