import numpy as np
from astropy import units as u
from astropy.table import QTable

from ctapipe.irf.cuts import EventQualitySelection


def test_optimization_result(tmp_path, irf_event_loader_test_config):
    from ctapipe.irf import (
        EventPreprocessor,
        OptimizationResult,
        ResultValidRange,
    )

    result_path = tmp_path / "result.h5"
    epp = EventPreprocessor(config=irf_event_loader_test_config)
    gh_cuts = QTable(
        data=[[0.2, 0.8, 1.5] * u.TeV, [0.8, 1.5, 10] * u.TeV, [0.82, 0.91, 0.88]],
        names=["low", "high", "cut"],
    )
    result = OptimizationResult(
        quality_selection=epp.event_selection,
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
    assert isinstance(loaded.quality_selection, EventQualitySelection)
    assert isinstance(loaded.valid_energy, ResultValidRange)
    assert isinstance(loaded.valid_offset, ResultValidRange)
    assert isinstance(loaded.gh_cuts, QTable)
    assert loaded.clf_prefix == "ExtraTreesClassifier"
