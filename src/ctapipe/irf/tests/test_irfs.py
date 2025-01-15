import astropy.units as u
from astropy.io.fits import BinTableHDU
from pyirf.simulations import SimulatedEventsInfo


def _check_boundaries_in_hdu(
    hdu: BinTableHDU,
    lo_vals: list,
    hi_vals: list,
    colnames: list[str] = ["THETA", "ENERG"],
):
    for col, val in zip(colnames, lo_vals):
        assert u.isclose(
            u.Quantity(hdu.data[f"{col}_LO"][0][0], hdu.columns[f"{col}_LO"].unit), val
        )
    for col, val in zip(colnames, hi_vals):
        assert u.isclose(
            u.Quantity(hdu.data[f"{col}_HI"][0][-1], hdu.columns[f"{col}_HI"].unit), val
        )


def test_make_2d_bkg(irf_events_table):
    from ctapipe.irf import BackgroundRate2dMaker

    bkg_maker = BackgroundRate2dMaker(
        fov_offset_n_bins=3,
        fov_offset_max=3 * u.deg,
        reco_energy_n_bins_per_decade=7,
        reco_energy_max=155 * u.TeV,
    )

    bkg_hdu = bkg_maker(events=irf_events_table, obs_time=1 * u.s)
    # min 7 bins per decade between 0.015 TeV and 155 TeV -> 7 * 4 + 1 = 29 bins
    assert bkg_hdu.data["BKG"].shape == (1, 3, 29)

    _check_boundaries_in_hdu(
        bkg_hdu, lo_vals=[0 * u.deg, 0.015 * u.TeV], hi_vals=[3 * u.deg, 155 * u.TeV]
    )


def test_make_2d_energy_migration(irf_events_table):
    from ctapipe.irf import EnergyDispersion2dMaker

    edisp_maker = EnergyDispersion2dMaker(
        fov_offset_n_bins=3,
        fov_offset_max=3 * u.deg,
        true_energy_n_bins_per_decade=7,
        true_energy_max=155 * u.TeV,
        energy_migration_n_bins=20,
        energy_migration_min=0.1,
        energy_migration_max=10,
    )
    edisp_hdu = edisp_maker(events=irf_events_table, spatial_selection_applied=False)
    # min 7 bins per decade between 0.015 TeV and 155 TeV -> 7 * 4 + 1 = 29 bins
    assert edisp_hdu.data["MATRIX"].shape == (1, 3, 20, 29)

    _check_boundaries_in_hdu(
        edisp_hdu,
        lo_vals=[0 * u.deg, 0.015 * u.TeV, 0.1],
        hi_vals=[3 * u.deg, 155 * u.TeV, 10],
        colnames=["THETA", "ENERG", "MIGRA"],
    )


def test_make_2d_eff_area(irf_events_table):
    from ctapipe.irf import EffectiveArea2dMaker

    eff_area_maker = EffectiveArea2dMaker(
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
    eff_area_hdu = eff_area_maker(
        events=irf_events_table,
        spatial_selection_applied=False,
        signal_is_point_like=False,
        sim_info=sim_info,
    )
    # min 7 bins per decade between 0.015 TeV and 155 TeV -> 7 * 4 + 1 = 29 bins
    assert eff_area_hdu.data["EFFAREA"].shape == (1, 3, 29)

    _check_boundaries_in_hdu(
        eff_area_hdu,
        lo_vals=[0 * u.deg, 0.015 * u.TeV],
        hi_vals=[3 * u.deg, 155 * u.TeV],
    )

    # point like data -> only 1 fov offset bin
    eff_area_hdu = eff_area_maker(
        events=irf_events_table,
        spatial_selection_applied=False,
        signal_is_point_like=True,
        sim_info=sim_info,
    )
    assert eff_area_hdu.data["EFFAREA"].shape == (1, 1, 29)


def test_make_3d_psf(irf_events_table):
    from ctapipe.irf import Psf3dMaker

    psf_maker = Psf3dMaker(
        fov_offset_n_bins=3,
        fov_offset_max=3 * u.deg,
        true_energy_n_bins_per_decade=7,
        true_energy_max=155 * u.TeV,
        source_offset_n_bins=110,
        source_offset_max=2 * u.deg,
    )
    psf_hdu = psf_maker(events=irf_events_table)
    # min 7 bins per decade between 0.015 TeV and 155 TeV -> 7 * 4 + 1 = 29 bins
    assert psf_hdu.data["RPSF"].shape == (1, 110, 3, 29)

    _check_boundaries_in_hdu(
        psf_hdu,
        lo_vals=[0 * u.deg, 0.015 * u.TeV, 0 * u.deg],
        hi_vals=[3 * u.deg, 155 * u.TeV, 2 * u.deg],
        colnames=["THETA", "ENERG", "RAD"],
    )
