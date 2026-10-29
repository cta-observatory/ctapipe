import yaml
from traitlets.config.loader import Config

from ctapipe.core import run_tool
from ctapipe.tools.estimate_nsb import EstimateNSBlevelTool
from ctapipe.utils import get_dataset_path


def test_estimate_nsb_tool_simulation(tmp_path):
    """Test EstimateNSBlevelTool execution on simulation data."""
    output_path = tmp_path / "nsb_params.yaml"

    input = get_dataset_path(
        "gamma_20deg_180deg_run000001___cta-prod6-2156m-LaPalma-dark.simtel.zst"
    )
    nsb_file = get_dataset_path("only_nsb_LaPalma_Alpha__nsb_x0.25.simtel.zst")

    config = Config(
        {
            "EstimateNSBlevelTool": {
                "quantile_threshold": 0.9,
                "mod_zscore_th": 3.5,
                "min_npixels_valid": 1000,
                "min_nped_events": 1,  # Input only contains 2 events for tel_id=12
                "image_extractor_type": "FixedWindowSum",
                "nsb_range": [
                    1,
                    5,
                ],  # We can only increase NSB up to 5 for the current statistics in nsb_file
            },
            "FixedWindowSum": {"window_width": 12, "window_shift": 6},
        }
    )

    ret = run_tool(
        EstimateNSBlevelTool(config=config),
        argv=[
            f"--input-file={input}",
            f"--nsb-file={nsb_file}",
            f"--sim-file={input}",
            f"--output={output_path}",
            "--overwrite",
        ],
    )

    assert ret == 0
    assert output_path.exists()

    with open(output_path, "r") as f:
        data = yaml.safe_load(f)

    assert "exact_nsb_level" in data
    assert "nsb_level" in data
    assert "mode_nsb_level" in data
    assert len(data["exact_nsb_level"]) > 0
    assert len(data["nsb_level"]) > 0

    expected_nsb_level = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
        9: 0,
        10: 0,
        11: 0,
        12: 1,
        13: 0,
    }
    for key in expected_nsb_level.keys():
        assert data["nsb_level"][key] == expected_nsb_level[key]

    assert data["mode_nsb_level"] == 0
