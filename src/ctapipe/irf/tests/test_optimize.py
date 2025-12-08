import astropy.units as u
import numpy as np
import pytest
from astropy.table import QTable

from ctapipe.core import QualityQuery, non_abstract_children
from ctapipe.irf.optimize import CutOptimizerBase, PointSourceSensitivityOptimizer


@pytest.mark.parametrize("file_format", [".fits", ".fits.gz"])
def test_optimization_result(tmp_path, irf_event_loader_test_config, file_format):
    from ctapipe.io import DL2EventPreprocessor
    from ctapipe.irf import (
        OptimizationResult,
        ResultValidRange,
    )

    result_path = tmp_path / f"result{file_format}"
    epp = DL2EventPreprocessor(irf_event_loader_test_config)
    gh_cuts = QTable(
        data=[[0.2, 0.8, 1.5] * u.TeV, [0.8, 1.5, 10] * u.TeV, [0.82, 0.91, 0.88]],
        names=["low", "high", "cut"],
    )
    result = OptimizationResult(
        quality_query=epp.quality_query,
        gh_cuts=gh_cuts,
        clf_prefix="ExtraTreesClassifier",
        valid_energy_min=0.2 * u.TeV,
        valid_energy_max=10 * u.TeV,
        valid_offset_min=0 * u.deg,
        valid_offset_max=np.inf * u.deg,
        spatial_selection_table=None,
    )
    result.write(result_path)
    assert result_path.exists()

    loaded = OptimizationResult.read(result_path)
    assert isinstance(loaded, OptimizationResult)
    assert isinstance(loaded.quality_query, QualityQuery)
    assert isinstance(loaded.valid_energy, ResultValidRange)
    assert isinstance(loaded.valid_offset, ResultValidRange)
    assert isinstance(loaded.gh_cuts, QTable)
    assert loaded.clf_prefix == "ExtraTreesClassifier"


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
    from ctapipe.io import DL2EventLoader
    from ctapipe.irf import OptimizationResult, Spectra

    gamma_loader = DL2EventLoader(
        config=irf_event_loader_test_config,
        file=gamma_diffuse_full_reco_file,
        target_spectrum=Spectra.CRAB_HEGRA,
    )
    gamma_events, _, _ = gamma_loader.load_preselected_events(
        chunk_size=10000,
        obs_time=u.Quantity(50, u.h),
    )
    proton_loader = DL2EventLoader(
        config=irf_event_loader_test_config,
        file=proton_full_reco_file,
        target_spectrum=Spectra.IRFDOC_PROTON_SPECTRUM,
    )
    proton_events, _, _ = proton_loader.load_preselected_events(
        chunk_size=10000,
        obs_time=u.Quantity(50, u.h),
    )

    optimizer = Optimizer()
    result = optimizer(
        events={"signal": gamma_events, "background": proton_events},
        quality_query=gamma_loader.epp.quality_query,  # identical qualityquery for all particle types
        clf_prefix="ExtraTreesClassifier",
    )
    assert isinstance(result, OptimizationResult)
    assert result.clf_prefix == "ExtraTreesClassifier"
    assert result.valid_energy.min >= result.gh_cuts["low"][0]
    assert result.valid_energy.max <= result.gh_cuts["high"][-1]
    assert result.spatial_selection_table["cut"].unit == u.deg

    if isinstance(optimizer, PointSourceSensitivityOptimizer):
        assert result.multiplicity_cuts is not None
        assert len(result.multiplicity_cuts["low"]) == len(result.gh_cuts["low"])
