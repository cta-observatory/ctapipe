#!/usr/bin/env python3
"""
Functions to help adapt internal ctapipe data to astropy formats and conventions
"""
import os
from contextlib import ExitStack
from uuid import uuid4

import numpy as np
import tables
from astropy.table import Table, join
from astropy.time import Time

from .hdf5tableio import (
    DEFAULT_FILTERS,
    get_column_attrs,
    get_column_transforms,
    get_node_meta,
)
from .tableio import (
    EnumColumnTransform,
    QuantityColumnTransform,
    StringTransform,
    TimeColumnTransform,
)

__all__ = ["read_table", "write_table", "join_allow_empty"]


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
        if not isinstance(table, tables.Table):
            raise OSError(
                f"Node {path} is a {table.__class__.__name__}, must be a Table"
            )
        transforms, descriptions, meta = _parse_hdf5_attrs(table)

        if condition is None:
            array = table.read(start=start, stop=stop, step=step)
        else:
            array = table.read_where(
                condition=condition, start=start, stop=stop, step=step
            )

        astropy_table = table_cls(array, meta=meta, copy=False)
        for column, tr in transforms.items():
            if column not in astropy_table.colnames:
                continue

            # keep enums as integers, much easier to deal with in tables
            if isinstance(tr, EnumColumnTransform):
                continue

            astropy_table[column] = tr.inverse(astropy_table[column])

        for column, desc in descriptions.items():
            if column not in astropy_table.colnames:
                continue

            astropy_table[column].description = desc

        return astropy_table


def write_table(
    table,
    h5file,
    path,
    append=False,
    overwrite=False,
    mode="a",
    filters=DEFAULT_FILTERS,
):
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
        Whether to try to append to or replace an existing table
    overwrite: bool
        If table is already in file and overwrite and append are false,
        raise an error.
    mode: str
        If given a path for ``h5file``, it will be opened in this mode.
        See the docs of ``tables.open_file``.
    """
    copied = False
    parent, table_name = os.path.split(path)

    if append and overwrite:
        raise ValueError("overwrite and append are mutually exclusive")

    with ExitStack() as stack:
        if not isinstance(h5file, tables.File):
            h5file = stack.enter_context(tables.open_file(h5file, mode=mode))

        already_exists = path in h5file.root
        if already_exists:
            if overwrite and not append:
                h5file.remove_node(parent, table_name)
                already_exists = False

            elif not overwrite and not append:
                raise OSError(
                    f"Table {path} already exists in output file, use append or overwrite"
                )

        attrs = {}
        for pos, (colname, column) in enumerate(table.columns.items()):
            if hasattr(column, "description") and column.description is not None:
                attrs[f"CTAFIELD_{pos}_DESC"] = column.description

            if isinstance(column, Time):
                transform = TimeColumnTransform(scale="tai", format="mjd")
                attrs.update(transform.get_meta(pos))

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
                attrs.update(transform.get_meta(pos))

            elif column.unit is not None:
                transform = QuantityColumnTransform(column.unit)
                attrs.update(transform.get_meta(pos))

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
    column_attrs = get_column_attrs(table)
    descriptions = {
        col_name: attrs.get("DESC", "") for col_name, attrs in column_attrs.items()
    }
    transforms = get_column_transforms(column_attrs)
    meta = get_node_meta(table)
    return transforms, descriptions, meta


def join_allow_empty(left, right, keys, join_type="left", keep_order=False, **kwargs):
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

    sort_key = None
    if keep_order:
        sort_key = str(uuid4())
        if join_type == "left":
            left[sort_key] = np.arange(len(left))
        elif join_type == "right":
            right[sort_key] = np.arange(len(left))
        else:
            raise ValueError("keep_order is only supported for left and right joins")

    joined = join(left, right, keys, join_type=join_type, **kwargs)
    if sort_key is not None:
        joined.sort(sort_key)
        del joined[sort_key]

    return joined
