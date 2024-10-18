import json

import astropy.units as u
import pytest
from astropy.table import QTable

from ctapipe.core import QualityQuery, run_tool


@pytest.mark.parametrize("point_like", (True, False))
def test_cuts_optimization(
    gamma_diffuse_full_reco_file,
    proton_full_reco_file,
    irf_event_loader_test_config,
    tmp_path,
    point_like,
):
    from ctapipe.irf import (
        OptimizationResult,
        ResultValidRange,
    )
    from ctapipe.tools.optimize_event_selection import IrfEventSelector

    output_path = tmp_path / "cuts.fits"
    config_path = tmp_path / "config.json"
    with config_path.open("w") as f:
        json.dump(irf_event_loader_test_config, f)

    argv = [
        f"--gamma-file={gamma_diffuse_full_reco_file}",
        f"--proton-file={proton_full_reco_file}",
        # Use diffuse gammas weighted to electron spectrum as stand-in
        f"--electron-file={gamma_diffuse_full_reco_file}",
        f"--output={output_path}",
        f"--config={config_path}",
    ]
    if point_like:
        argv.append("--point-like")

    ret = run_tool(IrfEventSelector(), argv=argv)
    assert ret == 0

    result = OptimizationResult.read(output_path)
    assert isinstance(result, OptimizationResult)
    assert isinstance(result.precuts, QualityQuery)
    assert isinstance(result.valid_energy, ResultValidRange)
    assert isinstance(result.valid_offset, ResultValidRange)
    assert isinstance(result.gh_cuts, QTable)
    assert result.clf_prefix == "ExtraTreesClassifier"
    assert "cut" in result.gh_cuts.colnames
    if point_like:
        assert isinstance(result.theta_cuts, QTable)
        assert "cut" in result.theta_cuts.colnames

    for c in ["low", "center", "high"]:
        assert c in result.gh_cuts.colnames
        assert result.gh_cuts[c].unit == u.TeV
        if point_like:
            assert c in result.theta_cuts.colnames
            assert result.theta_cuts[c].unit == u.TeV
