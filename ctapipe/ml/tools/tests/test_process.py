from ctapipe.core import run_tool
from ctapipe.io import read_table
import json


def test_process_apply_energy(tmp_path, energy_regressor_path):
    from ctapipe.tools.process import ProcessorTool

    output = tmp_path / "gamma_prod5.dl2_energy.h5"

    config_path = tmp_path / "config.json"

    config = {
        "ProcessorTool": {
            "stereo_combiner_configs": [
                {
                    "type": "StereoMeanCombiner",
                    "combine_property": "energy",
                    "algorithm": "ExtraTreesRegressor",
                    "weights": "konrad",
                }
            ]
        }
    }

    with config_path.open("w") as f:
        json.dump(config, f)

    argv = [
        "--input=dataset://gamma_prod5.simtel.zst",
        f"--output={output}",
        "--write-images",
        "--write-stereo-shower",
        "--write-mono-shower",
        f"--energy-regressor={energy_regressor_path}",
        f"--config={config_path}",
    ]
    assert run_tool(ProcessorTool(), argv=argv, cwd=tmp_path) == 0

    print(read_table(output, "/dl2/event/telescope/energy/ExtraTreesRegressor/tel_004"))
    print(read_table(output, "/dl2/event/subarray/energy/ExtraTreesRegressor"))
