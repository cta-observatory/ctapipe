import astropy.units as u
import numpy as np
import pytest
from astropy.table import QTable

from ctapipe.core import non_abstract_children
from ctapipe.irf import EventLoader, Spectra
from ctapipe.irf.optimize import CutOptimizerBase


def test_optimization_result_store(tmp_path, irf_events_loader_test_config):
    from ctapipe.irf import (
        EventPreProcessor,
        OptimizationResult,
        OptimizationResultStore,
        ResultValidRange,
    )

    result_path = tmp_path / "result.h5"
    epp = EventPreProcessor(irf_events_loader_test_config)
    store = OptimizationResultStore(epp)

    with pytest.raises(
        ValueError,
        match="The results of this object have not been properly initialised",
    ):
        store.write(result_path)

    gh_cuts = QTable(
        data=[[0.2, 0.8, 1.5] * u.TeV, [0.8, 1.5, 10] * u.TeV, [0.82, 0.91, 0.88]],
        names=["low", "high", "cut"],
    )
    store.set_result(
        gh_cuts=gh_cuts,
        valid_energy=[0.2 * u.TeV, 10 * u.TeV],
        valid_offset=[0 * u.deg, np.inf * u.deg],
        clf_prefix="ExtraTreesClassifier",
        theta_cuts=None,
    )
    store.write(result_path)
    assert result_path.exists()

    result = store.read(result_path)
    assert isinstance(result, OptimizationResult)
    assert isinstance(result.valid_energy, ResultValidRange)
    assert isinstance(result.valid_offset, ResultValidRange)
    assert isinstance(result.gh_cuts, QTable)
    assert result.gh_cuts.meta["CLFNAME"] == "ExtraTreesClassifier"


def test_gh_percentile_cut_calculator():
    from ctapipe.irf import GhPercentileCutCalculator

    calc = GhPercentileCutCalculator(
        target_percentile=75,
        min_counts=1,
        smoothing=-1,
    )
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

    calc = ThetaPercentileCutCalculator(
        target_percentile=75,
        min_counts=1,
        smoothing=-1,
    )
    cuts = calc.calculate_theta_cut(
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
    irf_events_loader_test_config,
):
    from ctapipe.irf import OptimizationResultStore

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

    optimizer = Optimizer()
    result = optimizer.optimize_cuts(
        signal=gamma_events,
        background=proton_events,
        alpha=0.2,
        precuts=gamma_loader.epp,  # identical precuts for all particle types
        clf_prefix="ExtraTreesClassifier",
        point_like=True,
    )
    assert isinstance(result, OptimizationResultStore)
    assert len(result._results) == 4
    assert result._results[1]["energy_min"] >= result._results[0]["low"][0]
    assert result._results[1]["energy_max"] <= result._results[0]["high"][-1]
    assert result._results[3]["cut"].unit == u.deg
