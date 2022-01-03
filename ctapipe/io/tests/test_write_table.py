# coding: utf-8
from astropy.time import Time
from astropy.table import Table
import astropy.units as u
import numpy as np


def test_write_table(tmp_path):
    from ctapipe.io.astropy_helpers import write_table, read_table

    table = Table(
        {
            "a": [1, 2, 3],
            "b": np.array([1, 2, 3], dtype=np.uint16),
            "speed": [2.0, 3.0, 4.2] * (u.m / u.s),
            "time": Time([58e3, 59e3, 60e3], format="mjd"),
            "name": ["a", "bb", "ccc"],
        }
    )

    table.meta["FOO"] = "bar"

    output_path = tmp_path = tmp_path / "table.h5"
    table_path = "/foo/bar"

    write_table(table, output_path, table_path)
    read = read_table(output_path, table_path)

    for name, column in table.columns.items():
        assert name in read.colnames, f"Column {name} not found in output file"
        assert (
            read.dtype[name] == table.dtype[name]
        ), f"Column {name} dtype different in output file"

        # time conversion is not lossless
        if name == "time":
            assert np.allclose(column.tai.mjd, read[name].tai.mjd)
        else:
            assert np.all(column == read[name]), f"Column {name} differs after reading"

    assert "FOO" in read.meta
    assert read.meta["FOO"] == "bar"

    # test we can append
    write_table(table, output_path, table_path, append=True)
    read = read_table(output_path, table_path)
    assert len(read) == 2 * len(table)

    # test we can overwrite
    write_table(table, output_path, table_path, append=False)
    assert len(read_table(output_path, table_path)) == len(table)
