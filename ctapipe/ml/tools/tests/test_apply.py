from ctapipe.containers import (
    EventIndexContainer,
    ParticleClassificationContainer,
    ReconstructedEnergyContainer,
)
from ctapipe.core import run_tool
from ctapipe.io import TableLoader, read_table


def test_apply_energy_regressor(
    energy_regressor_path,
    dl2_shower_geometry_file_lapalma,
    tmp_path,
):
    from ctapipe.ml.tools.apply import Apply

    input_path = dl2_shower_geometry_file_lapalma
    output_path = tmp_path / "energy.dl2.h5"

    ret = run_tool(
        Apply(),
        argv=[
            f"--input={input_path}",
            f"--output={output_path}",
            f"--regressor={energy_regressor_path}",
            "--Apply.StereoMeanCombiner.weights=konrad",
        ],
        raises=True,
    )
    assert ret == 0

    table = read_table(output_path, "/dl2/event/subarray/energy/ExtraTreesRegressor")
    for col in "obs_id", "event_id":
        assert table[col].description == EventIndexContainer.fields[col].description

    for name, field in ReconstructedEnergyContainer.fields.items():
        colname = f"ExtraTreesRegressor_{name}"
        assert colname in table.colnames
        assert table[colname].description == field.description

    loader = TableLoader(output_path, load_dl2=True)
    events = loader.read_subarray_events()
    assert "ExtraTreesRegressor_energy" in events.colnames
    assert "ExtraTreesRegressor_energy_uncert" in events.colnames
    assert "ExtraTreesRegressor_is_valid" in events.colnames
    assert "ExtraTreesRegressor_telescopes" in events.colnames

    events = loader.read_telescope_events()
    assert "ExtraTreesRegressor_energy" in events.colnames
    assert "ExtraTreesRegressor_energy_uncert" in events.colnames
    assert "ExtraTreesRegressor_is_valid" in events.colnames
    assert "ExtraTreesRegressor_telescopes" in events.colnames

    assert "ExtraTreesRegressor_tel_energy" in events.colnames
    assert "ExtraTreesRegressor_tel_is_valid" in events.colnames


def test_apply_particle_classifier(
    particle_classifier_path,
    dl2_shower_geometry_file_lapalma,
    tmp_path,
):
    from ctapipe.ml.tools.apply import Apply

    input_path = dl2_shower_geometry_file_lapalma
    output_path = tmp_path / "particle.dl2.h5"

    ret = run_tool(
        Apply(),
        argv=[
            f"--input={input_path}",
            f"--output={output_path}",
            f"--classifier={particle_classifier_path}",
            "--Apply.StereoMeanCombiner.weights=konrad",
        ],
    )
    assert ret == 0

    table = read_table(
        output_path, "/dl2/event/subarray/classification/ExtraTreesClassifier"
    )
    for col in "obs_id", "event_id":
        assert table[col].description == EventIndexContainer.fields[col].description

    for name, field in ParticleClassificationContainer.fields.items():
        colname = f"ExtraTreesClassifier_{name}"
        assert colname in table.colnames
        assert table[colname].description == field.description

    loader = TableLoader(output_path, load_dl2=True)
    events = loader.read_subarray_events()
    assert "ExtraTreesClassifier_prediction" in events.colnames
    assert "ExtraTreesClassifier_telescopes" in events.colnames
    assert "ExtraTreesClassifier_is_valid" in events.colnames
    assert "ExtraTreesClassifier_goodness_of_fit" in events.colnames

    events = loader.read_telescope_events()
    assert "ExtraTreesClassifier_prediction" in events.colnames
    assert "ExtraTreesClassifier_telescopes" in events.colnames
    assert "ExtraTreesClassifier_is_valid" in events.colnames
    assert "ExtraTreesClassifier_goodness_of_fit" in events.colnames

    assert "ExtraTreesClassifier_tel_prediction" in events.colnames
    assert "ExtraTreesClassifier_tel_is_valid" in events.colnames


def test_apply_both(
    energy_regressor_path,
    particle_classifier_path,
    dl2_shower_geometry_file_lapalma,
    tmp_path,
):
    from ctapipe.ml.tools.apply import Apply

    input_path = dl2_shower_geometry_file_lapalma
    output_path = tmp_path / "particle-and-energy.dl2.h5"

    ret = run_tool(
        Apply(),
        argv=[
            f"--input={input_path}",
            f"--output={output_path}",
            f"--classifier={particle_classifier_path}",
            f"--regressor={energy_regressor_path}",
            "--Apply.StereoMeanCombiner.weights=konrad",
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
