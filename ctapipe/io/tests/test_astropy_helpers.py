#!/usr/bin/env python3
import numpy as np
from astropy import units as u
import tables
import pytest

from ctapipe.containers import ReconstructedEnergyContainer
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
