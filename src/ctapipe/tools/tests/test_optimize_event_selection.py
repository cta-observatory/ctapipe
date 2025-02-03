import logging

import astropy.units as u
import pytest
from astropy.table import QTable

from ctapipe.core import QualityQuery, run_tool

pytest.importorskip("pyirf")


def test_cuts_optimization(
    gamma_diffuse_full_reco_file,
    proton_full_reco_file,
    event_loader_config_path,
    tmp_path,
):
    from ctapipe.irf import (
        OptimizationResult,
        ResultValidRange,
    )
    from ctapipe.tools.optimize_event_selection import EventSelectionOptimizer

    output_path = tmp_path / "cuts.fits"

    argv = [
        f"--gamma-file={gamma_diffuse_full_reco_file}",
        f"--proton-file={proton_full_reco_file}",
        # Use diffuse gammas weighted to electron spectrum as stand-in
        f"--electron-file={gamma_diffuse_full_reco_file}",
        f"--output={output_path}",
        f"--config={event_loader_config_path}",
    ]
    ret = run_tool(EventSelectionOptimizer(), argv=argv)
    assert ret == 0

    result = OptimizationResult.read(output_path)
    assert isinstance(result, OptimizationResult)
    assert isinstance(result.quality_query, QualityQuery)
    assert isinstance(result.valid_energy, ResultValidRange)
    assert isinstance(result.valid_offset, ResultValidRange)
    assert isinstance(result.gh_cuts, QTable)
    assert result.clf_prefix == "ExtraTreesClassifier"
    assert "cut" in result.gh_cuts.colnames
    assert isinstance(result.spatial_selection_table, QTable)
    assert "cut" in result.spatial_selection_table.colnames

    for c in ["low", "center", "high"]:
        assert c in result.gh_cuts.colnames
        assert result.gh_cuts[c].unit == u.TeV
        assert c in result.spatial_selection_table.colnames
        assert result.spatial_selection_table[c].unit == u.TeV


def test_cuts_opt_no_electrons(
    gamma_diffuse_full_reco_file,
    proton_full_reco_file,
    event_loader_config_path,
    tmp_path,
):
    from ctapipe.tools.optimize_event_selection import EventSelectionOptimizer

    output_path = tmp_path / "cuts.fits"
    logpath = tmp_path / "test_cuts_opt_no_electrons.log"
    logger = logging.getLogger("ctapipe.tools.optimize_event_selection")
    logger.addHandler(logging.FileHandler(logpath))

    ret = run_tool(
        EventSelectionOptimizer(),
        argv=[
            f"--gamma-file={gamma_diffuse_full_reco_file}",
            f"--proton-file={proton_full_reco_file}",
            f"--output={output_path}",
            f"--config={event_loader_config_path}",
            f"--log-file={logpath}",
        ],
    )
    assert ret == 0
    assert "Optimizing cuts without electron file." in logpath.read_text()


def test_cuts_opt_only_gammas(
    gamma_diffuse_full_reco_file, event_loader_config_path, tmp_path
):
    from ctapipe.tools.optimize_event_selection import EventSelectionOptimizer

    output_path = tmp_path / "cuts.fits"

    with pytest.raises(
        ValueError,
        match=(
            "Need a proton file for cut optimization using "
            "PointSourceSensitivityOptimizer"
        ),
    ):
        run_tool(
            EventSelectionOptimizer(),
            argv=[
                f"--gamma-file={gamma_diffuse_full_reco_file}",
                f"--output={output_path}",
                f"--config={event_loader_config_path}",
            ],
            raises=True,
        )

    ret = run_tool(
        EventSelectionOptimizer(),
        argv=[
            f"--gamma-file={gamma_diffuse_full_reco_file}",
            f"--output={output_path}",
            f"--config={event_loader_config_path}",
            "--EventSelectionOptimizer.optimization_algorithm=PercentileCuts",
        ],
    )
    assert ret == 0
    assert output_path.exists()
