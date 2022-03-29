#!/usr/bin/env python3
"""
Functions to help adapt internal ctapipe data to astropy formats and conventions
"""
import os
from contextlib import ExitStack

import tables
from astropy.table import Table, join
from astropy.time import Time
import numpy as np

from .tableio import (
    FixedPointColumnTransform,
    QuantityColumnTransform,
    TimeColumnTransform,
    StringTransform,
)
from .hdf5tableio import DEFAULT_FILTERS, get_hdf5_attr


__all__ = ["read_table", "join_allow_empty"]


def read_table(
    h5file, path, start=None, stop=None, step=None, condition=None, table_cls=Table
) -> Table:
    """Read a table from an HDF5 file

    This reads a table written in the ctapipe format table as an `astropy.table.Table`
    object, inversing the column transformations units.

    This uses the same conventions as the `~ctapipe.io.HDF5TableWriter`,
    with the exception of Enums, that will remain as integers.

    (start, stop, step) behave like python slices.

    Parameters
    ----------
    h5file: Union[str, Path, tables.file.File]
        input filename or PyTables file handle
    path: str
        path to table in the file
    start: int or None
        if given, this is the first row to be loaded
    stop: int or None
        if given, this is the last row to be loaded (not inclusive)
    step: int or None
        step between rows.
    condition: str
        A numexpr expression to only load rows fulfilling this condition.
        For example, use "hillas_length > 0" to only load rows where the
        hillas length is larger than 0 (so not nan and not 0).
        Ignored when reading tables that were written using astropy.

    Returns
    -------
    astropy.table.Table:
        table in Astropy Format

    """

    with ExitStack() as stack:

        if not isinstance(h5file, tables.File):
            h5file = stack.enter_context(tables.open_file(h5file))

        # check if the table was written using astropy table io, if yes
        # just use astropy
        is_astropy = f"{path}.__table_column_meta__" in h5file.root
        if is_astropy:
            sl = slice(start, stop, step)
            return table_cls.read(h5file.filename, path)[sl]

        # support leaving out the leading '/' for consistency with other
        # methods
        path = os.path.join("/", path)
        table = h5file.get_node(path)
        transforms, descriptions, meta = _parse_hdf5_attrs(table)

        if condition is None:
            array = table.read(start=start, stop=stop, step=step)
        else:
            array = table.read_where(
                condition=condition, start=start, stop=stop, step=step
            )

        astropy_table = table_cls(array, meta=meta, copy=False)
        for column, tr in transforms.items():
            astropy_table[column] = tr.inverse(astropy_table[column])

        for column, desc in descriptions.items():
            astropy_table[column].description = desc

        return astropy_table


def write_table(table, h5file, path, append=False, mode="a", filters=DEFAULT_FILTERS):
    """Write a table to an HDF5 file

    This writes a table in the ctapipe format into ``h5file``.

                attrs.update(transform.get_meta(name))

    Parameters
    ----------
    table: astropy.table.Table
        The table to be written.
    h5file: Union[str, Path, tables.file.File]
        input filename or PyTables file handle. If a PyTables file handle,
        must be opened writable.
    path: str
        dataset path inside the ``h5file``
    append: bool
        Wether to try to append to or replace an existing table
    mode: str
        If given a path for ``h5file``, it will be opened in this mode.
        See the docs of ``tables.open_file``.
    """
    copied = False

    with ExitStack() as stack:
        if not isinstance(h5file, tables.File):
            h5file = stack.enter_context(tables.open_file(h5file, mode=mode))

        attrs = {}
        for colname, column in table.columns.items():
            if hasattr(column, "description") and column.description is not None:
                attrs[f"{colname}_DESC"] = column.description

            if isinstance(column, Time):
                transform = TimeColumnTransform(scale="tai", format="mjd")
                attrs.update(transform.get_meta(colname))

                if copied is False:
                    table = table.copy()
                    copied = True

                table[colname] = transform(column)

            # TODO: use variable length strings as soon as tables supports them.
            # See PyTables/PyTables#48
            elif column.dtype.kind == "U":
                if copied is False:
                    table = table.copy()
                    copied = True

                table[colname] = np.array([s.encode("utf-8") for s in column])
                transform = StringTransform(table[colname].dtype.itemsize)
                attrs.update(transform.get_meta(colname))

            elif column.unit is not None:
                transform = QuantityColumnTransform(column.unit)
                attrs.update(transform.get_meta(colname))

        parent, table_name = os.path.split(path)

        already_exists = path in h5file.root

        if already_exists and not append:
            h5file.remove_node(parent, table_name)
            already_exists = False

        if not already_exists:
            h5_table = h5file.create_table(
                parent,
                table_name,
                filters=filters,
                expectedrows=len(table),
                createparents=True,
                obj=table.as_array(),
            )
        else:
            h5_table = h5file.get_node(path)
            h5_table.append(table.as_array())

        for key, val in table.meta.items():
            h5_table.attrs[key] = val

        for key, val in attrs.items():
            h5_table.attrs[key] = val


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
        elif attr.endswith("_TRANSFORM"):
            if table.attrs[attr] == "string":
                colname, _, _ = attr.rpartition("_TRANSFORM")
                column_transforms[colname] = StringTransform(
                    table.attrs[f"{colname}_MAXLEN"]
                )

        else:
            # need to convert to str() here so they are python strings, not
            # numpy strings
            value = table.attrs[attr]
            other_attrs[attr] = str(value) if isinstance(value, np.str_) else value

    return column_transforms, column_descriptions, other_attrs


def join_allow_empty(left, right, keys, join_type="left", **kwargs):
    """
    Join two astropy tables, allowing both sides to be empty tables.

    See https://github.com/astropy/astropy/issues/12012 for why
    this is necessary.

    This behaves as `~astropy.table.join`, with the only difference of
    allowing empty tables to be joined.
    """

    left_empty = len(left) == 0
    right_empty = len(right) == 0

    if join_type == "inner":
        if left_empty:
            return left.copy()
        if right_empty:
            return right.copy()

    elif join_type == "left":
        if left_empty or right_empty:
            return left.copy()

    elif join_type == "right":
        if left_empty or right_empty:
            return right.copy()

    elif join_type == "outer":
        if left_empty:
            return right.copy()

        if right_empty:
            return left.copy()

    return join(left, right, keys, join_type=join_type, **kwargs)
