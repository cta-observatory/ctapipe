#!/usr/bin/env python3
import numpy as np
from astropy import units as u
import tables
import pytest
from astropy.time import Time

from ctapipe.containers import ReconstructedEnergyContainer, TelescopeTriggerContainer
from ctapipe.io import HDF5TableWriter
from ctapipe.io.astropy_helpers import h5_table_to_astropy


def test_h5_table_to_astropy(tmp_path):

    # write a simple hdf5 file using

    container = ReconstructedEnergyContainer()
    filename = tmp_path / "test_astropy_table.h5"

    with HDF5TableWriter(filename) as writer:
        for energy in np.logspace(1, 2, 10) * u.TeV:
            container.energy = energy
            writer.write("events", container)

    # try opening the result
    table = h5_table_to_astropy(filename, "/events")

    assert "energy" in table.columns
    assert table["energy"].unit == u.TeV
    assert "CTAPIPE_VERSION" in table.meta
    assert table["energy"].description is not None

    # test using a string
    table = h5_table_to_astropy(str(filename), "/events")

    # test write the table back out to some other format:
    table.write(tmp_path / "test_output.ecsv")
    table.write(tmp_path / "test_output.fits.gz")

    # test using a file handle
    with tables.open_file(filename) as handle:
        table = h5_table_to_astropy(handle, "/events")

    # test a bad input
    with pytest.raises(ValueError):
        table = h5_table_to_astropy(12345, "/events")


def test_h5_table_to_astropy_time(tmp_path):
    t0 = Time("2020-01-01T20:00:00.0")
    times = t0 + np.arange(10) * u.s

    # use table writer to write test file
    filename = tmp_path / "test_astropy_table.h5"
    with HDF5TableWriter(filename) as writer:
        for t in times:
            container = TelescopeTriggerContainer(time=t, n_trigger_pixels=10)
            writer.write("events", container)

    # check reading in the times works as expected
    table = h5_table_to_astropy(filename, "/events")
    assert isinstance(table["time"], Time)
    assert np.allclose(times.tai.mjd, table["time"].tai.mjd)


def test_transforms(tmp_path):
    path = tmp_path / "test_trans.hdf5"

    data = np.array([100, 110], dtype="int16").view([("waveform", "int16")])
    print(data)

    with tables.open_file(path, "w") as f:
        f.create_table("/data", "test", obj=data, createparents=True)
        f.root.data.test.attrs["waveform_TRANSFORM_SCALE"] = 100.0
        f.root.data.test.attrs["waveform_TRANSFORM_OFFSET"] = 200
        f.root.data.test.attrs["waveform_TRANSFORM_DTYPE"] = "float64"

    table = h5_table_to_astropy(path, "/data/test")

    assert np.all(table["waveform"] == [-1.0, -0.9])
