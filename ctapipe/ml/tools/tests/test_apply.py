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
    from ctapipe.ml.tools.apply import ApplyModels

    input_path = dl2_shower_geometry_file_lapalma
    output_path = tmp_path / "energy.dl2.h5"

    ret = run_tool(
        ApplyModels(),
        argv=[
            f"--input={input_path}",
            f"--output={output_path}",
            f"--energy-regressor={energy_regressor_path}",
            "--ApplyModels.StereoMeanCombiner.weights=konrad",
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
    assert "ExtraTreesRegressor_goodness_of_fit" in events.colnames

    events = loader.read_telescope_events()
    assert "ExtraTreesRegressor_energy" in events.colnames
    assert "ExtraTreesRegressor_energy_uncert" in events.colnames
    assert "ExtraTreesRegressor_is_valid" in events.colnames
    assert "ExtraTreesRegressor_telescopes" in events.colnames
    assert "ExtraTreesRegressor_goodness_of_fit" in events.colnames

    assert "ExtraTreesRegressor_tel_energy" in events.colnames
    assert "ExtraTreesRegressor_tel_is_valid" in events.colnames


def test_apply_particle_classifier(
    particle_classifier_path,
    dl2_shower_geometry_file_lapalma,
    tmp_path,
):
    from ctapipe.ml.tools.apply import ApplyModels

    input_path = dl2_shower_geometry_file_lapalma
    output_path = tmp_path / "particle.dl2.h5"

    ret = run_tool(
        ApplyModels(),
        argv=[
            f"--input={input_path}",
            f"--output={output_path}",
            f"--particle-classifier={particle_classifier_path}",
            "--ApplyModels.StereoMeanCombiner.weights=konrad",
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


def test_apply_disp_reconstructor(
    disp_reconstructor_paths,
    dl2_shower_geometry_file_lapalma,
    tmp_path,
):
    from ctapipe.ml.tools.apply import ApplyModels

    input_path = dl2_shower_geometry_file_lapalma
    output_path = tmp_path / "disp.dl2.h5"

    ret = run_tool(
        ApplyModels(),
        argv=[
            f"--input={input_path}",
            f"--output={output_path}",
            f"--disp-regressor={disp_reconstructor_paths[0]}",
            f"--sign-classifier={disp_reconstructor_paths[1]}",
            "--ApplyModels.StereoMeanCombiner.weights=konrad",
        ],
    )
    assert ret == 0

    loader = TableLoader(output_path, load_dl2=True)
    events = loader.read_subarray_events()
    assert "ExtraTreesRegressor_ExtraTreesClassifier_alt" in events.colnames
    assert "ExtraTreesRegressor_ExtraTreesClassifier_alt_uncert" in events.colnames
    assert "ExtraTreesRegressor_ExtraTreesClassifier_az" in events.colnames
    assert "ExtraTreesRegressor_ExtraTreesClassifier_az_uncert" in events.colnames
    assert "ExtraTreesRegressor_ExtraTreesClassifier_is_valid" in events.colnames
    assert "ExtraTreesRegressor_ExtraTreesClassifier_goodness_of_fit" in events.colnames
    assert "ExtraTreesRegressor_ExtraTreesClassifier_tel_ids" in events.colnames

    events = loader.read_telescope_events()
    assert "ExtraTreesRegressor_ExtraTreesClassifier_alt" in events.colnames
    assert "ExtraTreesRegressor_ExtraTreesClassifier_alt_uncert" in events.colnames
    assert "ExtraTreesRegressor_ExtraTreesClassifier_az" in events.colnames
    assert "ExtraTreesRegressor_ExtraTreesClassifier_az_uncert" in events.colnames
    assert "ExtraTreesRegressor_ExtraTreesClassifier_is_valid" in events.colnames
    assert "ExtraTreesRegressor_ExtraTreesClassifier_goodness_of_fit" in events.colnames
    assert "ExtraTreesRegressor_ExtraTreesClassifier_tel_ids" in events.colnames

    assert "ExtraTreesRegressor_ExtraTreesClassifier_alt_mono" in events.colnames
    assert "ExtraTreesRegressor_ExtraTreesClassifier_az_mono" in events.colnames
    assert "ExtraTreesRegressor_ExtraTreesClassifier_is_valid_mono" in events.colnames
    assert "ExtraTreesRegressor_norm" in events.colnames
    assert "ExtraTreesRegressor_norm_is_valid" in events.colnames
    assert "ExtraTreesClassifier_sign" in events.colnames
    assert "ExtraTreesClassifier_sign_is_valid" in events.colnames


def test_apply_all(
    energy_regressor_path,
    particle_classifier_path,
    disp_reconstructor_paths,
    dl2_shower_geometry_file_lapalma,
    tmp_path,
):
    from ctapipe.ml.tools.apply import ApplyModels

    input_path = dl2_shower_geometry_file_lapalma
    output_path = tmp_path / "particle-and-energy-and-disp.dl2.h5"

    ret = run_tool(
        ApplyModels(),
        argv=[
            f"--input={input_path}",
            f"--output={output_path}",
            f"--particle-classifier={particle_classifier_path}",
            f"--energy-regressor={energy_regressor_path}",
            f"--disp-regressor={disp_reconstructor_paths[0]}",
            f"--sign-classifier={disp_reconstructor_paths[1]}",
            "--ApplyModels.StereoMeanCombiner.weights=konrad",
        ],
    )
    assert ret == 0

    loader = TableLoader(output_path, load_dl2=True)

    events = loader.read_subarray_events()
    assert "ExtraTreesRegressor_energy" in events.colnames
    assert "ExtraTreesClassifier_prediction" in events.colnames
    assert "ExtraTreesRegressor_ExtraTreesClassifier_alt" in events.colnames
    assert "ExtraTreesRegressor_ExtraTreesClassifier_az" in events.colnames

    events = loader.read_telescope_events()
    assert "ExtraTreesRegressor_energy" in events.colnames
    assert "ExtraTreesClassifier_prediction" in events.colnames
    assert "ExtraTreesRegressor_ExtraTreesClassifier_alt" in events.colnames
    assert "ExtraTreesRegressor_ExtraTreesClassifier_az" in events.colnames
