import astropy.units as u
import numpy as np
import pytest
from astropy.table import QTable, vstack
from pyirf.simulations import SimulatedEventsInfo

from ctapipe.irf import EventPreProcessor


@pytest.fixture(scope="session")
def irf_events_table():
    N1 = 1000
    N2 = 100
    N = N1 + N2
    epp = EventPreProcessor()
    tab = epp.make_empty_table()
    # Add fake weight column
    tab.add_column((), name="weight")
    units = {c: tab[c].unit for c in tab.columns}

    empty = np.zeros((len(tab.columns), N)) * np.nan
    e_tab = QTable(data=empty.T, names=tab.colnames, units=units)
    # Setting values following pyirf test in pyirf/irf/tests/test_background.py
    e_tab["reco_energy"] = np.append(np.full(N1, 1), np.full(N2, 2)) * u.TeV
    e_tab["true_energy"] = np.append(np.full(N1, 0.9), np.full(N2, 2.1)) * u.TeV
    e_tab["reco_source_fov_offset"] = (
        np.append(np.full(N1, 0.1), np.full(N2, 0.05)) * u.deg
    )
    e_tab["true_source_fov_offset"] = (
        np.append(np.full(N1, 0.11), np.full(N2, 0.04)) * u.deg
    )

    ev = vstack([e_tab, tab], join_type="exact", metadata_conflicts="silent")
    return ev


def test_make_2d_bkg(irf_events_table):
    from ctapipe.irf import BackgroundRate2dMaker

    bkgMkr = BackgroundRate2dMaker(
        fov_offset_n_bins=3,
        fov_offset_max=3 * u.deg,
        reco_energy_n_bins_per_decade=7,
        reco_energy_max=155 * u.TeV,
    )

    bkg_hdu = bkgMkr.make_bkg_hdu(events=irf_events_table, obs_time=1 * u.s)
    # min 7 bins per decade between 0.015 TeV and 155 TeV -> 7 * 4 + 1 = 29 bins
    assert bkg_hdu.data["BKG"].shape == (1, 3, 29)

    for col, val in zip(["THETA_LO", "ENERG_LO"], [0 * u.deg, 0.015 * u.TeV]):
        assert u.isclose(
            u.Quantity(bkg_hdu.data[col][0][0], bkg_hdu.columns[col].unit), val
        )

    for col, val in zip(["THETA_HI", "ENERG_HI"], [3 * u.deg, 155 * u.TeV]):
        assert u.isclose(
            u.Quantity(bkg_hdu.data[col][0][-1], bkg_hdu.columns[col].unit), val
        )


def test_make_2d_energy_migration(irf_events_table):
    from ctapipe.irf import EnergyMigration2dMaker

    migMkr = EnergyMigration2dMaker(
        fov_offset_n_bins=3,
        fov_offset_max=3 * u.deg,
        true_energy_n_bins_per_decade=7,
        true_energy_max=155 * u.TeV,
        energy_migration_n_bins=20,
        energy_migration_min=0.1,
        energy_migration_max=10,
    )
    mig_hdu = migMkr.make_edisp_hdu(events=irf_events_table, point_like=False)
    # min 7 bins per decade between 0.015 TeV and 155 TeV -> 7 * 4 + 1 = 29 bins
    assert mig_hdu.data["MATRIX"].shape == (1, 3, 20, 29)

    for col, val in zip(
        ["THETA_LO", "ENERG_LO", "MIGRA_LO"], [0 * u.deg, 0.015 * u.TeV, 0.1]
    ):
        assert u.isclose(
            u.Quantity(mig_hdu.data[col][0][0], mig_hdu.columns[col].unit), val
        )

    for col, val in zip(
        ["THETA_HI", "ENERG_HI", "MIGRA_HI"], [3 * u.deg, 155 * u.TeV, 10]
    ):
        assert u.isclose(
            u.Quantity(mig_hdu.data[col][0][-1], mig_hdu.columns[col].unit), val
        )


def test_make_2d_eff_area(irf_events_table):
    from ctapipe.irf import EffectiveArea2dMaker

    effAreaMkr = EffectiveArea2dMaker(
        fov_offset_n_bins=3,
        fov_offset_max=3 * u.deg,
        true_energy_n_bins_per_decade=7,
        true_energy_max=155 * u.TeV,
    )
    sim_info = SimulatedEventsInfo(
        n_showers=3000,
        energy_min=0.01 * u.TeV,
        energy_max=10 * u.TeV,
        max_impact=1000 * u.m,
        spectral_index=-1.9,
        viewcone_min=0 * u.deg,
        viewcone_max=10 * u.deg,
    )
    eff_area_hdu = effAreaMkr.make_aeff_hdu(
        events=irf_events_table,
        point_like=False,
        signal_is_point_like=False,
        sim_info=sim_info,
    )
    # min 7 bins per decade between 0.015 TeV and 155 TeV -> 7 * 4 + 1 = 29 bins
    assert eff_area_hdu.data["EFFAREA"].shape == (1, 3, 29)

    for col, val in zip(["THETA_LO", "ENERG_LO"], [0 * u.deg, 0.015 * u.TeV]):
        assert u.isclose(
            u.Quantity(eff_area_hdu.data[col][0][0], eff_area_hdu.columns[col].unit),
            val,
        )

    for col, val in zip(["THETA_HI", "ENERG_HI"], [3 * u.deg, 155 * u.TeV]):
        assert u.isclose(
            u.Quantity(eff_area_hdu.data[col][0][-1], eff_area_hdu.columns[col].unit),
            val,
        )

    # point like data -> only 1 fov offset bin
    eff_area_hdu = effAreaMkr.make_aeff_hdu(
        events=irf_events_table,
        point_like=False,
        signal_is_point_like=True,
        sim_info=sim_info,
    )
    assert eff_area_hdu.data["EFFAREA"].shape == (1, 1, 29)


def test_make_3d_psf(irf_events_table):
    from ctapipe.irf import Psf3dMaker

    psfMkr = Psf3dMaker(
        fov_offset_n_bins=3,
        fov_offset_max=3 * u.deg,
        true_energy_n_bins_per_decade=7,
        true_energy_max=155 * u.TeV,
        source_offset_n_bins=110,
        source_offset_max=2 * u.deg,
    )
    psf_hdu = psfMkr.make_psf_hdu(events=irf_events_table)
    # min 7 bins per decade between 0.015 TeV and 155 TeV -> 7 * 4 + 1 = 29 bins
    assert psf_hdu.data["RPSF"].shape == (1, 110, 3, 29)

    for col, val in zip(
        ["THETA_LO", "ENERG_LO", "RAD_LO"], [0 * u.deg, 0.015 * u.TeV, 0 * u.deg]
    ):
        assert u.isclose(
            u.Quantity(psf_hdu.data[col][0][0], psf_hdu.columns[col].unit),
            val,
        )

    for col, val in zip(
        ["THETA_HI", "ENERG_HI", "RAD_HI"], [3 * u.deg, 155 * u.TeV, 2 * u.deg]
    ):
        assert u.isclose(
            u.Quantity(psf_hdu.data[col][0][-1], psf_hdu.columns[col].unit),
            val,
        )
