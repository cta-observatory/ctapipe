#!/usr/bin/env python3

from pathlib import Path

import tables
from astropy.table import QTable
from astropy.units import Unit


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
    if not isinstance(h5file, tables.file.File) and isinstance(h5file, (str, Path)):
        h5file = tables.open_file(h5file)
        should_close_file = True
    else:
        raise ValueError(
            f"expected a string, Path, or PyTables "
            f"filehandle for argument 'h5file', got {h5file}"
        )

    table = h5file.get_node(path)

    column_units = {}  # mapping of colname to unit
    for attr in table.attrs._f_list():
        if attr.endswith("_UNIT"):
            colname = attr[:-5]
            column_units[colname] = table.attrs[attr]

    astropy_table = QTable(table[:])

    for column, unit in column_units.items():
        astropy_table[column].unit = Unit(unit)

    if should_close_file:
        h5file.close()

    return astropy_table
