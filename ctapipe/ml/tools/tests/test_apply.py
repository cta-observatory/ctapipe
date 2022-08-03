from ctapipe.core import run_tool
from ctapipe.io import TableLoader


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

    loader = TableLoader(output_path, load_dl2=True)
    events = loader.read_subarray_events()
    assert "ExtraTreesRegressor_energy" in events.colnames
    assert "ExtraTreesRegressor_energy_uncert" in events.colnames
    assert "ExtraTreesRegressor_is_valid" in events.colnames
    assert "ExtraTreesRegressor_tel_ids" in events.colnames

    events = loader.read_telescope_events()
    assert "ExtraTreesRegressor_energy" in events.colnames
    assert "ExtraTreesRegressor_energy_uncert" in events.colnames
    assert "ExtraTreesRegressor_is_valid" in events.colnames
    assert "ExtraTreesRegressor_tel_ids" in events.colnames

    assert "ExtraTreesRegressor_energy_mono" in events.colnames
    assert "ExtraTreesRegressor_is_valid_mono" in events.colnames


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

    loader = TableLoader(output_path, load_dl2=True)
    events = loader.read_subarray_events()
    assert "ExtraTreesClassifier_prediction" in events.colnames
    assert "ExtraTreesClassifier_tel_ids" in events.colnames
    assert "ExtraTreesClassifier_is_valid" in events.colnames
    assert "ExtraTreesClassifier_goodness_of_fit" in events.colnames

    events = loader.read_telescope_events()
    assert "ExtraTreesClassifier_prediction" in events.colnames
    assert "ExtraTreesClassifier_tel_ids" in events.colnames
    assert "ExtraTreesClassifier_is_valid" in events.colnames
    assert "ExtraTreesClassifier_goodness_of_fit" in events.colnames

    assert "ExtraTreesClassifier_prediction_mono" in events.colnames
    assert "ExtraTreesClassifier_is_valid_mono" in events.colnames


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
