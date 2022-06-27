from ctapipe.core import run_tool
from ctapipe.io import read_table
import json


def test_process_apply_energy(tmp_path, energy_regressor_path):
    from ctapipe.tools.process import ProcessorTool
    from ctapipe.io import SimTelEventSource

    output = tmp_path / "gamma_prod5.dl2_energy.h5"

    config_path = tmp_path / "config.json"

    input_url = "dataset://gamma_prod5.simtel.zst"

    with SimTelEventSource(input_url) as s:
        subarray = s.subarray

    allowed_tels = subarray.get_tel_ids_for_type(
        "LST_LST_LSTCam"
    ) + subarray.get_tel_ids_for_type("MST_MST_NectarCam")

    config = {
        "ProcessorTool": {
            "EventSource": {
                "allowed_tels": allowed_tels,
            },
            "stereo_combiner_configs": [
                {
                    "type": "StereoMeanCombiner",
                    "combine_property": "energy",
                    "algorithm": "ExtraTreesRegressor",
                    "weights": "konrad",
                }
            ],
        }
    }

    with config_path.open("w") as f:
        json.dump(config, f)

    argv = [
        f"--input={input_url}",
        f"--output={output}",
        "--write-images",
        "--write-showers",
        f"--energy-regressor={energy_regressor_path}",
        f"--config={config_path}",
    ]
    assert run_tool(ProcessorTool(), argv=argv, cwd=tmp_path) == 0

    print(read_table(output, "/dl2/event/telescope/energy/ExtraTreesRegressor/tel_004"))
    print(read_table(output, "/dl2/event/subarray/energy/ExtraTreesRegressor"))


def test_process_apply_classification(tmp_path, particle_classifier_path):
    from ctapipe.tools.process import ProcessorTool
    from ctapipe.io import SimTelEventSource

    output = tmp_path / "gamma_prod5.dl2_energy.h5"

    config_path = tmp_path / "config.json"

    input_url = "dataset://gamma_prod5.simtel.zst"

    with SimTelEventSource(input_url) as s:
        subarray = s.subarray

    allowed_tels = subarray.get_tel_ids_for_type(
        "LST_LST_LSTCam"
    ) + subarray.get_tel_ids_for_type("MST_MST_NectarCam")

    config = {
        "ProcessorTool": {
            "EventSource": {
                "allowed_tels": allowed_tels,
            },
            "stereo_combiner_configs": [
                {
                    "type": "StereoMeanCombiner",
                    "combine_property": "classification",
                    "algorithm": "ExtraTreesClassifier",
                }
            ],
        }
    }

    with config_path.open("w") as f:
        json.dump(config, f)

    argv = [
        f"--input={input_url}",
        f"--output={output}",
        "--write-images",
        "--write-showers",
        f"--particle-classifier={particle_classifier_path}",
        f"--config={config_path}",
    ]
    assert run_tool(ProcessorTool(), argv=argv, cwd=tmp_path) == 0

    print(
        read_table(
            output, "/dl2/event/telescope/classification/ExtraTreesClassifier/tel_004"
        )
    )
    print(read_table(output, "/dl2/event/subarray/classification/ExtraTreesClassifier"))
