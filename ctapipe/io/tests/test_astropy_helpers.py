#!/usr/bin/env python3
import numpy as np
from astropy import units as u
from ctapipe.io import HDF5TableWriter
from ctapipe.io.astropy_helpers import h5_table_to_astropy
from ctapipe.containers import ReconstructedEnergyContainer


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
