#!/usr/bin/env python3
import warnings
import numpy as np
from astropy import units as u
import tables
import pytest
from astropy.time import Time
from astropy.table import Table
from astropy.utils.diff import report_diff_values

from astropy.io.fits.verify import VerifyWarning

from ctapipe.core import Container, Field
from ctapipe.containers import ReconstructedEnergyContainer, TelescopeTriggerContainer
from ctapipe.io import HDF5TableWriter
from ctapipe.io.astropy_helpers import read_table
from io import StringIO



def assert_table_equal(a, b):
    '''
    Assert that two astropy tables are the same.

    Compares two tables using the astropy diff utility
    and use the report as error message in case they don't match
    '''
    msg = StringIO()
    msg.write('\n')
    valid = report_diff_values(a, b, fileobj=msg)
    msg.seek(0)
    assert valid, msg.read()


def test_read_table(tmp_path):
    # write a simple hdf5 file using
    container = ReconstructedEnergyContainer()
    filename = tmp_path / "test_astropy_table.h5"

    with HDF5TableWriter(filename) as writer:
        for energy in np.logspace(1, 2, 10) * u.TeV:
            container.energy = energy
            writer.write("events", container)

    # try opening the result
    table = read_table(filename, "/events")

    assert "energy" in table.columns
    assert table["energy"].unit == u.TeV
    assert "CTAPIPE_VERSION" in table.meta
    assert table["energy"].description is not None

    # test using a string
    table = read_table(str(filename), "/events")

    # test write the table back out to some other format:
    table.write(tmp_path / "test_output.ecsv")
    with warnings.catch_warnings():
        # ignore warnings about too long keywords stored using HIERARCH
        warnings.simplefilter("ignore", VerifyWarning)
        table.write(tmp_path / "test_output.fits.gz")

    # test using a file handle
    with tables.open_file(filename) as handle:
        table = read_table(handle, "/events")

    # test a bad input
    with pytest.raises(TypeError):
        table = read_table(12345, "/events")


def test_read_table_slicing(tmp_path):
    filename = tmp_path / "test_slicing.h5"

    # write a simple hdf5 file using
    class Data(Container):
        index = Field(0)
        value = Field(0.0)

    rng = np.random.default_rng(0)
    values = rng.normal(size=100)
    index = np.arange(len(values))

    with HDF5TableWriter(filename) as writer:
        for i, value in zip(index, values):
            container = Data(index=i, value=value)
            writer.write("events", container)

    # try opening the result
    table = read_table(filename, "/events", start=50)
    assert len(table) == 50
    assert np.all(table["index"] == index[50:])
    assert np.all(table["value"] == values[50:])

    table = read_table(filename, "/events", stop=50)
    assert len(table) == 50
    assert np.all(table["index"] == index[:50])
    assert np.all(table["value"] == values[:50])

    table = read_table(filename, "/events", start=10, stop=30)
    assert len(table) == 20
    assert np.all(table["index"] == index[10:30])
    assert np.all(table["value"] == values[10:30])

    table = read_table(filename, "/events", step=5)
    assert len(table) == 20
    assert np.all(table["index"] == index[::5])
    assert np.all(table["value"] == values[::5])


def test_read_table_time(tmp_path):
    t0 = Time("2020-01-01T20:00:00.0")
    times = t0 + np.arange(10) * u.s

    # use table writer to write test file
    filename = tmp_path / "test_astropy_table.h5"
    with HDF5TableWriter(filename) as writer:
        for t in times:
            container = TelescopeTriggerContainer(time=t, n_trigger_pixels=10)
            writer.write("events", container)

    # check reading in the times works as expected
    table = read_table(filename, "/events")
    assert isinstance(table["time"], Time)
    assert np.allclose(times.tai.mjd, table["time"].tai.mjd)


def test_transforms(tmp_path):
    path = tmp_path / "test_trans.hdf5"

    data = np.array([100, 110], dtype="int16").view([("waveform", "int16")])

    with tables.open_file(path, "w") as f:
        f.create_table("/data", "test", obj=data, createparents=True)
        f.root.data.test.attrs["waveform_TRANSFORM_SCALE"] = 100.0
        f.root.data.test.attrs["waveform_TRANSFORM_OFFSET"] = 200
        f.root.data.test.attrs["waveform_TRANSFORM_DTYPE"] = "float64"

    table = read_table(path, "/data/test")

    assert np.all(table["waveform"] == [-1.0, -0.9])


def test_file_closed(tmp_path):
    """Test read_table closes the file even when an exception happens during read"""

    path = tmp_path / "empty.hdf5"
    with tables.open_file(path, "w"):
        pass

    with pytest.raises(tables.NoSuchNodeError):
        read_table(path, "/foo")

    # test we can open the file for writing, fails if read_table did not close
    # the file
    with tables.open_file(path, "w"):
        pass


def test_condition(tmp_path):
    # write a simple hdf5 file using

    container = ReconstructedEnergyContainer()
    filename = tmp_path / "test_astropy_table.h5"

    with HDF5TableWriter(filename) as writer:
        for energy in [np.nan, 100, np.nan, 50, -1.0] * u.TeV:
            container.energy = energy
            writer.write("events", container)

    # try opening the result
    table = read_table(filename, "/events", condition="energy > 0")
    assert len(table) == 2
    assert np.all(table["energy"] == [100, 50] * u.TeV)


def test_read_table_astropy(tmp_path):
    """Test that ctapipe.io.read_table can also read a table written Table.write"""
    table = Table(
        {
            "a": [1, 2, 3],
            "b": np.array([1, 2, 3], dtype=np.uint16),
            "speed": [2.0, 3.0, 4.2] * (u.m / u.s),
        }
    )

    path = tmp_path / "test.h5"
    table.write(path, "/group/table", serialize_meta=True)
    read = read_table(path, "/group/table")
    assert_table_equal(table, read)
