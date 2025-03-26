"""Tests for the write_table function"""
import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table
from astropy.time import Time


@pytest.mark.parametrize(
    ("time_format", "tolerance"),
    [
        ("mjd", 1 * u.us),
        ("ctao_high_res", 0.01 * u.ns),
    ],
)
def test_write_table(tmp_path, time_format, tolerance):
    """Test write_table"""
    from ctapipe.io.astropy_helpers import read_table, write_table

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
    table["speed"].description = "Speed of stuff"

    output_path = tmp_path = tmp_path / "table.h5"
    table_path = "/foo/bar"

    write_table(table, output_path, table_path, time_format=time_format)
    read = read_table(output_path, table_path)

    for name, column in table.columns.items():
        assert name in read.colnames, f"Column {name} not found in output file"
        assert (
            read.dtype[name] == table.dtype[name]
        ), f"Column {name} dtype different in output file"

        # time conversion is not lossless
        if name == "time":
            np.testing.assert_array_less(
                np.abs((column - read[name]).to(u.ns)), tolerance
            )
        else:
            assert np.all(column == read[name]), f"Column {name} differs after reading"

    assert "FOO" in read.meta
    assert read.meta["FOO"] == "bar"
    assert read["speed"].description == "Speed of stuff"

    # test error for already existing table
    with pytest.raises(OSError):
        write_table(table, output_path, table_path)

    # test we can append
    write_table(table, output_path, table_path, append=True, time_format=time_format)
    read = read_table(output_path, table_path)
    assert len(read) == 2 * len(table)

    # test we can overwrite
    write_table(table, output_path, table_path, overwrite=True)
    assert len(read_table(output_path, table_path)) == len(table)


def test_invalid_input():
    """Test invalid path argument raises"""
    from ctapipe.io.astropy_helpers import write_table

    with pytest.raises(TypeError):
        invalid_path = 5
        write_table(Table({"a": [1, 2, 3]}), invalid_path, "/temp")
