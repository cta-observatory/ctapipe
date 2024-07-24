import astropy.units as u
import numpy as np

from ctapipe.irf import EventsLoader, Spectra


def test_gh_percentile_cut_calculator():
    from ctapipe.irf import GhPercentileCutCalculator

    calc = GhPercentileCutCalculator()
    calc.target_percentile = 75
    calc.min_counts = 1
    calc.smoothing = -1
    cuts = calc.calculate_gh_cut(
        gammaness=np.array([0.1, 0.6, 0.45, 0.98, 0.32, 0.95, 0.25, 0.87]),
        reco_energy=[0.17, 0.36, 0.47, 0.22, 1.2, 5, 4.2, 9.1] * u.TeV,
        reco_energy_bins=[0, 1, 10] * u.TeV,
    )
    assert len(cuts) == 2
    assert np.isclose(cuts["cut"][0], 0.3625)
    assert np.isclose(cuts["cut"][1], 0.3025)
    assert calc.smoothing is None


def test_theta_percentile_cut_calculator():
    from ctapipe.irf import ThetaPercentileCutCalculator

    calc = ThetaPercentileCutCalculator()
    calc.target_percentile = 75
    calc.min_counts = 1
    calc.smoothing = -1
    cuts = calc.calculate_theta_cut(
        theta=[0.1, 0.07, 0.21, 0.4, 0.03, 0.08, 0.11, 0.18] * u.deg,
        reco_energy=[0.17, 0.36, 0.47, 0.22, 1.2, 5, 4.2, 9.1] * u.TeV,
        reco_energy_bins=[0, 1, 10] * u.TeV,
    )
    assert len(cuts) == 2
    assert np.isclose(cuts["cut"][0], 0.2575 * u.deg)
    assert np.isclose(cuts["cut"][1], 0.1275 * u.deg)
    assert calc.smoothing is None


def test_percentile_cuts(gamma_diffuse_full_reco_file, irf_events_loader_test_config):
    from ctapipe.irf import OptimizationResultStore, PercentileCuts

    loader = EventsLoader(
        config=irf_events_loader_test_config,
        kind="gammas",
        file=gamma_diffuse_full_reco_file,
        target_spectrum=Spectra.CRAB_HEGRA,
    )
    events, _, _ = loader.load_preselected_events(
        chunk_size=10000,
        obs_time=u.Quantity(50, u.h),
    )
    optimizer = PercentileCuts()
    result = optimizer.optimize_cuts(
        signal=events,
        background=None,
        alpha=0.2,  # Default value in the tool, not used for PercentileCuts
        precuts=loader.epp,
        clf_prefix="ExtraTreesClassifier",
        point_like=True,
    )
    assert isinstance(result, OptimizationResultStore)
    assert len(result._results) == 4
    assert u.isclose(result._results[1]["energy_min"], result._results[0]["low"][0])
    assert u.isclose(result._results[1]["energy_max"], result._results[0]["high"][-1])
    assert result._results[3]["cut"].unit == u.deg
