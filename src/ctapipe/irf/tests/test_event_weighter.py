#!/usr/bin/env python3

import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable

from ctapipe.irf.event_weighter import RadialEventWeighter, SimpleEventWeighter


@pytest.fixture(scope="function")
def example_weight_table():
    table = QTable(
        dict(
            true_energy=[1.0, 2.0, 0.5, 0.2] * u.TeV,
            true_fov_offset=[0.1, 1.2, 2.2, 3.2] * u.deg,
        )
    )
    table.meta["VERSION"] = 1.0
    table.columns["true_energy"].description = "True energy of particle"
    return table


def test_simple_weights(example_weight_table):
    from ctapipe.irf.spectra import PowerLaw

    table = example_weight_table

    source_spectrum = PowerLaw(
        normalization=u.Quantity(0.00027, "TeV-1 s-1 sr-1 m-2"),
        index=-2.0,
        e_ref=1.0 * u.TeV,
    )

    weight1 = SimpleEventWeighter(
        source_spectrum=source_spectrum, target_spectrum_name="CRAB_HEGRA"
    )

    table_w1 = weight1(table)

    assert np.all(table_w1["weight"] > 0.0)
    assert np.all(table_w1["weight"] <= 1.0)


def test_flat_weighting(example_weight_table):
    """Check that if source and target spectra are the same,  we get 1.0."""
    from ctapipe.irf.spectra import Spectra, spectrum_from_name

    table = example_weight_table

    weight = SimpleEventWeighter(
        source_spectrum=spectrum_from_name(Spectra.CRAB_HEGRA),
        target_spectrum_name=Spectra.CRAB_HEGRA.name,
        is_diffuse=False,
    )

    w = weight(table)["weight"]

    assert np.allclose(w, 1.0)


def test_radial_weights(example_weight_table):
    from ctapipe.irf.spectra import PowerLaw

    table = example_weight_table

    source_spectrum = PowerLaw(
        normalization=u.Quantity(0.00027, "TeV-1 s-1 sr-1 m-2"),
        index=-2.0,
        e_ref=1.0 * u.TeV,
    )

    weight = RadialEventWeighter(
        source_spectrum=source_spectrum,
        target_spectrum_name="CRAB_HEGRA",
        fov_offset_min=0.0 * u.deg,
        fov_offset_max=5.0 * u.deg,
        fov_offset_n_bins=5,
    )

    assert np.allclose(weight.fov_offset_bins, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0] * u.deg)

    table_w = weight(table)

    assert "fov_offset_bin" in table_w.colnames
    assert "OFFSBINS" in table_w.meta
    assert table_w.meta["OFFSBINS"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    for ii in range(0, 7):
        mask = table_w["fov_offset_bin"] == ii

        if ii == 0:
            # there should be no 0th bin
            assert len(table_w[mask]) == 0
        elif ii == 6:
            # for outlier bin 6, weights should be all 0
            assert np.all(table_w["weight"][mask] == 0.0)
        else:
            # for bins 1-5,weights  should be positive in this case
            assert np.all(table_w["weight"][mask] >= 0.0)


def test_radial_weights_fits(example_weight_table, tmp_path):
    """Test round-tripping to FITS"""
    from astropy.table import Table

    from ctapipe.irf.spectra import PowerLaw

    table = example_weight_table

    source_spectrum = PowerLaw(
        normalization=u.Quantity(0.00027, "TeV-1 s-1 sr-1 m-2"),
        index=-2.0,
        e_ref=1.0 * u.TeV,
    )

    weight = RadialEventWeighter(
        source_spectrum=source_spectrum,
        target_spectrum_name="CRAB_HEGRA",
        fov_offset_min=0.0 * u.deg,
        fov_offset_max=5.0 * u.deg,
        fov_offset_n_bins=5,
    )

    table_w = weight(table)
    table_w.write(tmp_path / "test.fits")

    assert Table.read(tmp_path / "test.fits").meta["OFFSBINS"] == [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ]
