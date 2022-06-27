from ctapipe.core import run_tool
from ctapipe.io import TableLoader
import shutil


def test_apply_energy_regressor(
    energy_regressor_path,
    dl2_shower_geometry_file_lapalma,
    tmp_path,
):
    from ctapipe.ml.tools.apply_energy_regressor import ApplyEnergyRegressor

    input_path = tmp_path / dl2_shower_geometry_file_lapalma.name

    # create copy to not mutate common test file
    shutil.copy2(dl2_shower_geometry_file_lapalma, input_path)

    ret = run_tool(
        ApplyEnergyRegressor(),
        argv=[
            f"--input={input_path}",
            f"--model={energy_regressor_path}",
            "--ApplyEnergyRegressor.StereoMeanCombiner.weights=konrad",
        ],
    )
    assert ret == 0

    loader = TableLoader(input_path, load_dl2=True)
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
    from ctapipe.ml.tools.apply_particle_classifier import ApplyParticleIdClassifier

    input_path = tmp_path / dl2_shower_geometry_file_lapalma.name

    # create copy to not mutate common test file
    shutil.copy2(dl2_shower_geometry_file_lapalma, input_path)

    ret = run_tool(
        ApplyParticleIdClassifier(),
        argv=[
            f"--input={input_path}",
            f"--model={particle_classifier_path}",
            "--ApplyParticleIdClassifier.StereoMeanCombiner.weights=konrad",
        ],
    )
    assert ret == 0

    loader = TableLoader(input_path, load_dl2=True)
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
