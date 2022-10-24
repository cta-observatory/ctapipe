import json

from ctapipe.core import run_tool
from ctapipe.io import read_table


def test_process_apply_energy(
    tmp_path, energy_regressor_path, prod5_gamma_lapalma_simtel_path
):
    from ctapipe.tools.process import ProcessorTool

    output = tmp_path / "gamma_prod5.dl2_energy.h5"

    config_path = tmp_path / "config.json"

    input_url = prod5_gamma_lapalma_simtel_path

    # la palma alpha config
    allowed_tels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 19, 35]
    config = {
        "ProcessorTool": {
            "EventSource": {
                "allowed_tels": allowed_tels,
            },
            "ShowerProcessor": {
                "reconstructor_types": [
                    "HillasReconstructor",
                    "EnergyRegressor",
                ],
                "EnergyRegressor": {
                    "load_path": str(energy_regressor_path),
                },
            },
        }
    }

    with config_path.open("w") as f:
        json.dump(config, f)

    argv = [
        f"--input={input_url}",
        f"--output={output}",
        "--write-images",
        "--write-showers",
        f"--config={config_path}",
    ]
    assert run_tool(ProcessorTool(), argv=argv, cwd=tmp_path, raises=True) == 0

    events = read_table(output, "/dl2/event/subarray/energy/ExtraTreesRegressor")
    tel_events = read_table(
        output, "/dl2/event/telescope/energy/ExtraTreesRegressor/tel_004"
    )
    assert len(events) > 0
    assert len(tel_events) > 0


def test_process_apply_classification(
    tmp_path, particle_classifier_path, prod5_gamma_lapalma_simtel_path
):
    from ctapipe.tools.process import ProcessorTool

    output = tmp_path / "gamma_prod5.dl2_energy.h5"

    config_path = tmp_path / "config.json"

    input_url = prod5_gamma_lapalma_simtel_path

    allowed_tels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 19, 35]
    config = {
        "ProcessorTool": {
            "EventSource": {
                "allowed_tels": allowed_tels,
            },
            "ShowerProcessor": {
                "reconstructor_types": [
                    "HillasReconstructor",
                    "ParticleClassifier",
                ]
            },
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
    assert run_tool(ProcessorTool(), argv=argv, cwd=tmp_path, raises=True) == 0

    print(
        read_table(
            output, "/dl2/event/telescope/classification/ExtraTreesClassifier/tel_004"
        )
    )
    print(read_table(output, "/dl2/event/subarray/classification/ExtraTreesClassifier"))
