import json

import numpy as np

from ctapipe.core import run_tool
from ctapipe.io import read_table
from ctapipe.io.hdf5dataformat import (
    DL1_SUBARRAY_TRIGGER_TABLE,
    DL2_SUBARRAY_ENERGY_GROUP,
    DL2_SUBARRAY_GEOMETRY_GROUP,
    DL2_SUBARRAY_GROUP,
    DL2_TEL_ENERGY_GROUP,
    DL2_TEL_GEOMETRY_GROUP,
    DL2_TEL_GROUP,
)


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
        },
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

    events = read_table(output, f"{DL2_SUBARRAY_ENERGY_GROUP}/ExtraTreesRegressor")
    tel_events = read_table(
        output, f"{DL2_TEL_ENERGY_GROUP}/ExtraTreesRegressor/tel_004"
    )
    assert len(events) > 0
    assert len(tel_events) > 0


def test_process_apply_classification(
    tmp_path,
    energy_regressor_path,
    particle_classifier_path,
    prod5_gamma_lapalma_simtel_path,
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
        }
    }

    with config_path.open("w") as f:
        json.dump(config, f)

    argv = [
        f"--input={input_url}",
        f"--output={output}",
        f"--config={config_path}",
        "--write-images",
        "--write-showers",
        "--reconstructor=HillasReconstructor",
        "--reconstructor=EnergyRegressor",
        "--reconstructor=ParticleClassifier",
        f"--EnergyRegressor.load_path={energy_regressor_path}",
        f"--ParticleClassifier.load_path={particle_classifier_path}",
    ]
    tool = ProcessorTool()
    run_tool(tool, argv=argv, cwd=tmp_path, raises=True)

    tel_events = read_table(
        output, f"{DL2_TEL_GROUP}/particle_type/ExtraTreesClassifier/tel_004"
    )
    assert "ExtraTreesClassifier_tel_is_valid" in tel_events.colnames
    assert "ExtraTreesClassifier_tel_prediction" in tel_events.colnames

    events = read_table(
        output, f"{DL2_SUBARRAY_GROUP}/particle_type/ExtraTreesClassifier"
    )
    trigger = read_table(output, DL1_SUBARRAY_TRIGGER_TABLE)
    assert len(events) == len(trigger)
    assert "ExtraTreesClassifier_is_valid" in events.colnames
    assert "ExtraTreesClassifier_prediction" in events.colnames
    np.testing.assert_array_equal(events["obs_id"], trigger["obs_id"])
    np.testing.assert_array_equal(events["event_id"], trigger["event_id"])


def test_process_apply_disp(
    tmp_path,
    energy_regressor_path,
    disp_reconstructor_path,
    prod5_gamma_lapalma_simtel_path,
):
    from ctapipe.tools.process import ProcessorTool

    disp_reconstructor_path, _ = disp_reconstructor_path

    output = tmp_path / "gamma_prod5.dl2_disp.h5"

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
                    "EnergyRegressor",
                    "DispReconstructor",
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
        f"--config={config_path}",
        f"--EnergyRegressor.load_path={energy_regressor_path}",
        f"--DispReconstructor.load_path={disp_reconstructor_path}",
    ]

    tool = ProcessorTool()
    run_tool(tool, argv=argv, cwd=tmp_path, raises=True)

    tel_events = read_table(output, f"{DL2_TEL_GROUP}/disp/disp/tel_004")
    assert "disp_tel_parameter" in tel_events.colnames

    tel_events = read_table(output, f"{DL2_TEL_GEOMETRY_GROUP}/disp/tel_004")
    assert "disp_tel_alt" in tel_events.colnames
    assert "disp_tel_az" in tel_events.colnames
    assert "disp_tel_is_valid" in tel_events.colnames

    events = read_table(output, f"{DL2_SUBARRAY_GEOMETRY_GROUP}/disp")
    trigger = read_table(output, DL1_SUBARRAY_TRIGGER_TABLE)
    assert len(events) == len(trigger)
    assert "disp_alt" in events.colnames
    assert "disp_az" in events.colnames
    assert "disp_is_valid" in events.colnames
    np.testing.assert_array_equal(events["obs_id"], trigger["obs_id"])
    np.testing.assert_array_equal(events["event_id"], trigger["event_id"])
