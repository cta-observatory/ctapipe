import sys

import astropy.units as u
import pytest
from astropy.table import QTable

from ctapipe.irf.tests.test_irfs import _check_boundaries_in_hdu


def test_make_2d_energy_bias_res(irf_events_table):
    from ctapipe.irf import EnergyBiasResolution2dMaker

    bias_res_maker = EnergyBiasResolution2dMaker(
        fov_offset_n_bins=3,
        fov_offset_max=3 * u.deg,
        true_energy_n_bins_per_decade=7,
        true_energy_max=155 * u.TeV,
    )

    bias_res_hdu = bias_res_maker.make_bias_resolution_hdu(events=irf_events_table)
    # min 7 bins per decade between 0.015 TeV and 155 TeV -> 7 * 4 + 1 = 29 bins
    assert (
        bias_res_hdu.data["N_EVENTS"].shape
        == bias_res_hdu.data["BIAS"].shape
        == bias_res_hdu.data["RESOLUTION"].shape
        == (1, 3, 29)
    )
    _check_boundaries_in_hdu(
        bias_res_hdu,
        lo_vals=[0 * u.deg, 0.015 * u.TeV],
        hi_vals=[3 * u.deg, 155 * u.TeV],
    )


def test_make_2d_ang_res(irf_events_table):
    from ctapipe.irf import AngularResolution2dMaker

    ang_res_maker = AngularResolution2dMaker(
        fov_offset_n_bins=3,
        fov_offset_max=3 * u.deg,
        true_energy_n_bins_per_decade=7,
        true_energy_max=155 * u.TeV,
        reco_energy_n_bins_per_decade=6,
        reco_energy_min=0.03 * u.TeV,
    )

    ang_res_hdu = ang_res_maker.make_angular_resolution_hdu(events=irf_events_table)
    assert (
        ang_res_hdu.data["N_EVENTS"].shape
        == ang_res_hdu.data["ANGULAR_RESOLUTION"].shape
        == (1, 3, 23)
    )
    _check_boundaries_in_hdu(
        ang_res_hdu,
        lo_vals=[0 * u.deg, 0.03 * u.TeV],
        hi_vals=[3 * u.deg, 150 * u.TeV],
    )

    ang_res_maker.use_true_energy = True
    ang_res_hdu = ang_res_maker.make_angular_resolution_hdu(events=irf_events_table)
    assert (
        ang_res_hdu.data["N_EVENTS"].shape
        == ang_res_hdu.data["ANGULAR_RESOLUTION"].shape
        == (1, 3, 29)
    )
    _check_boundaries_in_hdu(
        ang_res_hdu,
        lo_vals=[0 * u.deg, 0.015 * u.TeV],
        hi_vals=[3 * u.deg, 155 * u.TeV],
    )


@pytest.mark.skipif(sys.version_info.minor > 11, reason="Pyirf+numpy 2.0 errors out")
def test_make_2d_sensitivity(
    gamma_diffuse_full_reco_file, proton_full_reco_file, irf_events_loader_test_config
):
    from ctapipe.irf import EventLoader, Sensitivity2dMaker, Spectra

    gamma_loader = EventLoader(
        config=irf_events_loader_test_config,
        kind="gammas",
        file=gamma_diffuse_full_reco_file,
        target_spectrum=Spectra.CRAB_HEGRA,
    )
    gamma_events, _, _ = gamma_loader.load_preselected_events(
        chunk_size=10000,
        obs_time=u.Quantity(50, u.h),
    )
    proton_loader = EventLoader(
        config=irf_events_loader_test_config,
        kind="protons",
        file=proton_full_reco_file,
        target_spectrum=Spectra.IRFDOC_PROTON_SPECTRUM,
    )
    proton_events, _, _ = proton_loader.load_preselected_events(
        chunk_size=10000,
        obs_time=u.Quantity(50, u.h),
    )

    sens_maker = Sensitivity2dMaker(
        fov_offset_n_bins=3,
        fov_offset_max=3 * u.deg,
        reco_energy_n_bins_per_decade=7,
        reco_energy_max=155 * u.TeV,
    )
    # Create a dummy theta cut since `pyirf.sensitivity.estimate_background`
    # needs a theta cut atm.
    theta_cuts = QTable()
    theta_cuts["center"] = 0.5 * (
        sens_maker.reco_energy_bins[:-1] + sens_maker.reco_energy_bins[1:]
    )
    theta_cuts["cut"] = sens_maker.fov_offset_max

    sens_hdu = sens_maker.make_sensitivity_hdu(
        signal_events=gamma_events,
        background_events=proton_events,
        theta_cut=theta_cuts,
        gamma_spectrum=Spectra.CRAB_HEGRA,
    )
    assert (
        sens_hdu.data["N_SIGNAL"].shape
        == sens_hdu.data["N_SIGNAL_WEIGHTED"].shape
        == sens_hdu.data["N_BACKGROUND"].shape
        == sens_hdu.data["N_BACKGROUND_WEIGHTED"].shape
        == sens_hdu.data["SIGNIFICANCE"].shape
        == sens_hdu.data["RELATIVE_SENSITIVITY"].shape
        == sens_hdu.data["FLUX_SENSITIVITY"].shape
        == (1, 3, 29)
    )
    _check_boundaries_in_hdu(
        sens_hdu,
        lo_vals=[0 * u.deg, 0.015 * u.TeV],
        hi_vals=[3 * u.deg, 155 * u.TeV],
    )
