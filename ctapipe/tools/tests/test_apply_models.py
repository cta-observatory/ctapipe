import numpy as np

from ctapipe.containers import EventIndexContainer, ReconstructedEnergyContainer
from ctapipe.core import run_tool
from ctapipe.io import TableLoader, read_table


def test_apply_energy_regressor(
    energy_regressor_path,
    dl2_shower_geometry_file_lapalma,
    tmp_path,
):
    from ctapipe.tools.apply_models import ApplyModels

    input_path = dl2_shower_geometry_file_lapalma
    output_path = tmp_path / "energy.dl2.h5"

    ret = run_tool(
        ApplyModels(),
        argv=[
            f"--input={input_path}",
            f"--output={output_path}",
            f"--energy-regressor={energy_regressor_path}",
            "--StereoMeanCombiner.weights=konrad",
        ],
        raises=True,
    )
    assert ret == 0

    prefix = "ExtraTreesRegressor"
    table = read_table(output_path, f"/dl2/event/subarray/energy/{prefix}")
    for col in "obs_id", "event_id":
        assert table[col].description == EventIndexContainer.fields[col].description

    for name, field in ReconstructedEnergyContainer.fields.items():
        colname = f"{prefix}_{name}"
        assert colname in table.colnames
        assert table[colname].description == field.description

    loader = TableLoader(output_path, load_dl2=True)
    events = loader.read_subarray_events()

    assert f"{prefix}_energy" in events.colnames
    assert f"{prefix}_energy_uncert" in events.colnames
    assert f"{prefix}_is_valid" in events.colnames
    assert f"{prefix}_telescopes" in events.colnames
    assert np.any(events[f"{prefix}_is_valid"])
    assert np.all(np.isfinite(events[f"{prefix}_energy"][events[f"{prefix}_is_valid"]]))

    events = loader.read_telescope_events()
    assert f"{prefix}_energy" in events.colnames
    assert f"{prefix}_energy_uncert" in events.colnames
    assert f"{prefix}_is_valid" in events.colnames
    assert f"{prefix}_telescopes" in events.colnames

    assert f"{prefix}_tel_energy" in events.colnames
    assert f"{prefix}_tel_is_valid" in events.colnames


def test_apply_both(
    energy_regressor_path,
    particle_classifier_path,
    dl2_shower_geometry_file_lapalma,
    tmp_path,
):
    from ctapipe.tools.apply_models import ApplyModels

    input_path = dl2_shower_geometry_file_lapalma
    output_path = tmp_path / "particle-and-energy.dl2.h5"

    ret = run_tool(
        ApplyModels(),
        argv=[
            f"--input={input_path}",
            f"--output={output_path}",
            f"--energy-regressor={energy_regressor_path}",
            f"--particle-classifier={particle_classifier_path}",
            "--StereoMeanCombiner.weights=konrad",
        ],
    )
    assert ret == 0

    loader = TableLoader(output_path, load_dl2=True)

    events = loader.read_subarray_events()
    assert "ExtraTreesRegressor_energy" in events.colnames
    assert "ExtraTreesClassifier_prediction" in events.colnames

    events = loader.read_telescope_events()
    assert "ExtraTreesClassifier_prediction" in events.colnames
    assert "ExtraTreesRegressor_energy" in events.colnames
