import astropy.units as u
import numpy as np
import pytest

from ctapipe.core import non_abstract_children
from ctapipe.irf.optimize.algorithm import CutOptimizerBase


def test_gh_percentile_cut_calculator():
    from ctapipe.irf import GhPercentileCutCalculator

    calc = GhPercentileCutCalculator(
        target_percentile=75,
        min_counts=1,
        smoothing=-1,
    )
    cuts = calc(
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

    calc = ThetaPercentileCutCalculator(
        target_percentile=75,
        min_counts=1,
        smoothing=-1,
    )
    cuts = calc(
        theta=[0.1, 0.07, 0.21, 0.4, 0.03, 0.08, 0.11, 0.18] * u.deg,
        reco_energy=[0.17, 0.36, 0.47, 0.22, 1.2, 5, 4.2, 9.1] * u.TeV,
        reco_energy_bins=[0, 1, 10] * u.TeV,
    )
    assert len(cuts) == 2
    assert np.isclose(cuts["cut"][0], 0.2575 * u.deg)
    assert np.isclose(cuts["cut"][1], 0.1275 * u.deg)
    assert calc.smoothing is None


@pytest.mark.parametrize("Optimizer", non_abstract_children(CutOptimizerBase))
def test_cut_optimizer(
    Optimizer,
    gamma_diffuse_full_reco_file,
    proton_full_reco_file,
    irf_event_loader_test_config,
):
    from ctapipe.irf import EventLoader, OptimizationResult, Spectra

    gamma_loader = EventLoader(
        file=gamma_diffuse_full_reco_file,
        target_spectrum=Spectra.CRAB_HEGRA,
        config=irf_event_loader_test_config,
    )
    gamma_events = gamma_loader.load_preselected_events(chunk_size=10000)
    proton_loader = EventLoader(
        file=proton_full_reco_file,
        target_spectrum=Spectra.IRFDOC_PROTON_SPECTRUM,
        config=irf_event_loader_test_config,
    )
    proton_events = proton_loader.load_preselected_events(chunk_size=10000)

    optimizer = Optimizer()
    result = optimizer(
        events={"signal": gamma_events, "background": proton_events},
        quality_query=gamma_loader.epp.event_selection,  # identical qualityquery for all particle types
        clf_prefix="ExtraTreesClassifier",
    )
    assert isinstance(result, OptimizationResult)
    assert result.clf_prefix == "ExtraTreesClassifier"
    assert result.valid_energy.min >= result.gh_cuts["low"][0]
    assert result.valid_energy.max <= result.gh_cuts["high"][-1]
    assert result.spatial_selection_table["cut"].unit == u.deg
