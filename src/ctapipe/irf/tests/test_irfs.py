import astropy.units as u
import numpy as np
import pytest
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


def _sim_info():
    return SimulatedEventsInfo(
        n_showers=3000,
        energy_min=0.01 * u.TeV,
        energy_max=10 * u.TeV,
        max_impact=1000 * u.m,
        spectral_index=-1.9,
        viewcone_min=0 * u.deg,
        viewcone_max=10 * u.deg,
    )


def test_make_2d_background(irf_events_table):
    from ctapipe.irf import BackgroundRate2dMaker

    background_maker = BackgroundRate2dMaker(
        fov_offset_n_bins=3,
        fov_offset_max=3 * u.deg,
        reco_energy_n_bins_per_decade=7,
        reco_energy_max=155 * u.TeV,
    )

    background_hdu = background_maker(events=irf_events_table, obs_time=1 * u.s)
    # min 7 bins per decade between 0.015 TeV and 155 TeV -> 7 * 4 + 1 = 29 bins
    assert background_hdu.data["BKG"].shape == (1, 3, 29)

    _check_boundaries_in_hdu(
        background_hdu,
        lo_vals=[0 * u.deg, 0.015 * u.TeV],
        hi_vals=[3 * u.deg, 155 * u.TeV],
    )


def test_make_2d_energy_migration(irf_events_table):
    from ctapipe.irf import EnergyDispersion2dMaker

    energy_dispersion_maker = EnergyDispersion2dMaker(
        fov_offset_n_bins=3,
        fov_offset_max=3 * u.deg,
        true_energy_n_bins_per_decade=7,
        true_energy_max=155 * u.TeV,
        energy_migration_n_bins=20,
        energy_migration_min=0.1,
        energy_migration_max=10,
    )
    energy_dispersion_hdu = energy_dispersion_maker(
        events=irf_events_table, spatial_selection_applied=False
    )
    # min 7 bins per decade between 0.015 TeV and 155 TeV -> 7 * 4 + 1 = 29 bins
    assert energy_dispersion_hdu.data["MATRIX"].shape == (1, 3, 20, 29)

    _check_boundaries_in_hdu(
        energy_dispersion_hdu,
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
    sim_info = _sim_info()
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


def test_make_3d_eff_area(irf_events_table):
    from ctapipe.irf import EffectiveArea3DMaker

    true_fov_lon = irf_events_table["true_source_fov_lon"].copy()
    true_fov_lat = irf_events_table["true_source_fov_lat"].copy()

    eff_area_maker = EffectiveArea3DMaker(
        fov_lon_n_bins=5,
        fov_lon_max=3 * u.deg,
        fov_lat_n_bins=5,
        fov_lat_max=3 * u.deg,
        true_energy_n_bins_per_decade=7,
        true_energy_max=155 * u.TeV,
    )
    sim_info = _sim_info()
    eff_area_hdu = eff_area_maker(
        events=irf_events_table,
        spatial_selection_applied=False,
        signal_is_point_like=False,
        sim_info=sim_info,
    )
    # min 7 bins per decade between 0.015 TeV and 155 TeV -> 7 * 4 + 1 = 29 bins
    assert eff_area_hdu.data["EFFAREA"].shape == (1, 5, 5, 29)

    _check_boundaries_in_hdu(
        eff_area_hdu,
        lo_vals=[-3 * u.deg, -3 * u.deg, 0.015 * u.TeV],
        hi_vals=[3 * u.deg, 3 * u.deg, 155 * u.TeV],
        colnames=["DETX", "DETY", "ENERG"],
    )

    assert eff_area_hdu.header["HDUCLAS4"] == "AEFF_3D"
    assert eff_area_hdu.header["HDUCLAS3"] == "FULL-ENCLOSURE"
    assert eff_area_hdu.header["HDUCLASS"] == "GADF"
    assert eff_area_hdu.header["HDUVERS"] == "0.4"
    assert eff_area_hdu.header["FOVALIGN"] == "ALTAZ"

    # the selection only affects HDUCLAS3, not the computed effective area
    eff_area_hdu = eff_area_maker(
        events=irf_events_table,
        spatial_selection_applied=False,
        signal_is_point_like=True,
        sim_info=sim_info,
    )
    assert eff_area_hdu.data["EFFAREA"].shape == (1, 5, 5, 29)
    assert eff_area_hdu.header["HDUCLAS3"] == "POINT-LIKE"

    eff_area_hdu = eff_area_maker(
        events=irf_events_table,
        spatial_selection_applied=True,
        signal_is_point_like=False,
        sim_info=sim_info,
    )
    assert eff_area_hdu.data["EFFAREA"].shape == (1, 5, 5, 29)
    assert eff_area_hdu.header["HDUCLAS3"] == "POINT-LIKE"

    # the maker must not modify the input table
    assert u.allclose(irf_events_table["true_source_fov_lon"], true_fov_lon)
    assert u.allclose(irf_events_table["true_source_fov_lat"], true_fov_lat)


@pytest.mark.parametrize(
    "lon, has_area",
    [(1 * u.deg, True), (-1 * u.deg, False)],
    ids=["native_positive_lon", "native_negative_lon"],
)
def test_make_3d_eff_area_gadf_sign(irf_events_table, lon, has_area):
    from ctapipe.irf import EffectiveArea3DMaker

    eff_area_maker = EffectiveArea3DMaker(
        fov_lon_n_bins=4,
        fov_lon_min=-4 * u.deg,
        fov_lon_max=0 * u.deg,
        fov_lat_n_bins=4,
        fov_lat_min=-1 * u.deg,
        fov_lat_max=1 * u.deg,
        true_energy_n_bins_per_decade=7,
        true_energy_max=155 * u.TeV,
    )
    # A square fov grid is required until pyirf 0.15 (which returns the fov
    # axes in (lat, lon) order for calculate_n_showers_3d_lonlat) is replaced;
    # a non-square grid test will be implemented with the next pyirf release.

    events = irf_events_table.copy()
    # the dl2 columns store the native fov lon; the maker flips the sign for GADF
    events["true_source_fov_lon"] = lon * np.ones(len(events))
    events["true_source_fov_lat"] = np.zeros(len(events)) * u.deg

    eff_area_hdu = eff_area_maker(
        events=events,
        spatial_selection_applied=False,
        signal_is_point_like=False,
        sim_info=_sim_info(),
    )

    eff_area = eff_area_hdu.data["EFFAREA"]
    total_area = np.nansum(np.where(np.isnan(eff_area), 0, eff_area))
    assert (total_area > 0) == has_area


def test_make_3d_psf(irf_events_table):
    from ctapipe.irf import PSF3DMaker

    psf_maker = PSF3DMaker(
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
