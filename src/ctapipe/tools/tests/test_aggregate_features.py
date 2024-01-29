import json

import pytest

from ctapipe.core import ToolConfigurationError, run_tool
from ctapipe.core.traits import TraitError
from ctapipe.io import TableLoader


def test_aggregate_features(dl2_shower_geometry_file_lapalma, tmp_path):
    from ctapipe.tools.aggregate_features import AggregateFeatures

    input_path = dl2_shower_geometry_file_lapalma
    output_path = tmp_path / "aggregated.dl1.h5"
    config_path = tmp_path / "config.json"

    with pytest.raises(
        TraitError, match="No DL1 image parameters to aggregate are specified."
    ):
        run_tool(
            AggregateFeatures(),
            argv=[
                f"--input={input_path}",
                f"--output={output_path}",
            ],
            raises=True,
        )

    config = {
        "FeatureAggregator": {
            "FeatureGenerator": {
                "features": [("hillas_abs_skewness", "abs(hillas_skewness)")]
            },
            "image_parameters": [
                ("hillas", "length"),
                ("timing", "slope"),
                ("HillasReconstructor", "tel_impact_distance"),
                ("hillas", "abs_skewness"),
            ],
        }
    }
    with config_path.open("w") as f:
        json.dump(config, f)

    # test "overwrite" works
    with pytest.raises(ToolConfigurationError, match="exists, but overwrite=False"):
        run_tool(
            AggregateFeatures(),
            argv=[
                f"--input={input_path}",
                f"--output={output_path}",
                f"--config={config_path}",
            ],
            raises=True,
        )

    ret = run_tool(
        AggregateFeatures(),
        argv=[
            f"--input={input_path}",
            f"--output={output_path}",
            f"--config={config_path}",
            "--overwrite",
        ],
        raises=True,
    )
    assert ret == 0

    with TableLoader(output_path) as loader:
        events = loader.read_subarray_events(
            dl1_aggregates=True,
            dl2=False,
            simulated=False,
        )
        for col in [
            "hillas_length",
            "timing_slope",
            "HillasReconstructor_tel_impact_distance",
            "hillas_abs_skewness",
        ]:
            for suffix in ["max", "min", "mean", "std"]:
                assert f"{col}_{suffix}" in events.colnames
