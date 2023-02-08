import numpy as np

from ctapipe.containers import (
    EventIndexContainer,
    ParticleClassificationContainer,
    ReconstructedEnergyContainer,
    ReconstructedGeometryContainer,
)
from ctapipe.core import run_tool
from ctapipe.io import TableLoader, read_table
from ctapipe.io.tests.test_table_loader import check_equal_array_event_order


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
            f"--reconstructor={energy_regressor_path}",
            "--StereoMeanCombiner.weights=konrad",
            "--chunk-size=5",  # small chunksize so we test multiple chunks for the test file
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

    with TableLoader(output_path, load_dl2=True) as loader:

        events = loader.read_subarray_events()
        assert f"{prefix}_energy" in events.colnames
        assert f"{prefix}_energy_uncert" in events.colnames
        assert f"{prefix}_is_valid" in events.colnames
        assert f"{prefix}_telescopes" in events.colnames
        assert np.any(events[f"{prefix}_is_valid"])
        assert np.all(
            np.isfinite(events[f"{prefix}_energy"][events[f"{prefix}_is_valid"]])
        )

        tel_events = loader.read_telescope_events()
        assert f"{prefix}_energy" in tel_events.colnames
        assert f"{prefix}_energy_uncert" in tel_events.colnames
        assert f"{prefix}_is_valid" in tel_events.colnames
        assert f"{prefix}_telescopes" in events.colnames

        assert f"{prefix}_tel_energy" in tel_events.colnames
        assert f"{prefix}_tel_is_valid" in tel_events.colnames
        assert "hillas_intensity" in tel_events.colnames

    trigger = read_table(output_path, "/dl1/event/subarray/trigger")
    energy = read_table(output_path, "/dl2/event/subarray/energy/ExtraTreesRegressor")

    check_equal_array_event_order(trigger, energy)


def test_apply_all(
    energy_regressor_path,
    particle_classifier_path,
    disp_reconstructor_path,
    dl2_shower_geometry_file_lapalma,
    tmp_path,
):
    from ctapipe.tools.apply_models import ApplyModels

    input_path = dl2_shower_geometry_file_lapalma
    output_path = tmp_path / "particle-and-energy-and-disp.dl2.h5"

    ret = run_tool(
        ApplyModels(),
        argv=[
            f"--input={input_path}",
            f"--output={output_path}",
            f"--reconstructor={energy_regressor_path}",
            f"--reconstructor={particle_classifier_path}",
            f"--reconstructor={disp_reconstructor_path}",
            "--no-dl1-parameters",
            "--no-true-parameters",
            "--StereoMeanCombiner.weights=konrad",
            "--chunk-size=5",  # small chunksize so we test multiple chunks for the test file
        ],
        raises=True,
    )
    assert ret == 0

    prefix_clf = "ExtraTreesClassifier"
    table = read_table(output_path, f"/dl2/event/subarray/classification/{prefix_clf}")
    for col in "obs_id", "event_id":
        assert table[col].description == EventIndexContainer.fields[col].description

    for name, field in ParticleClassificationContainer.fields.items():
        colname = f"{prefix_clf}_{name}"
        assert colname in table.colnames
        assert table[colname].description == field.description

    prefix_disp = "disp"
    table = read_table(output_path, f"/dl2/event/subarray/geometry/{prefix_disp}")
    for col in "obs_id", "event_id":
        assert table[col].description == EventIndexContainer.fields[col].description

    for name, field in ReconstructedGeometryContainer.fields.items():
        colname = f"{prefix_disp}_{name}"
        assert colname in table.colnames
        assert table[colname].description == field.description

    with TableLoader(output_path, load_dl2=True) as loader:
        events = loader.read_subarray_events()
        assert f"{prefix_clf}_prediction" in events.colnames
        assert f"{prefix_clf}_telescopes" in events.colnames
        assert f"{prefix_clf}_is_valid" in events.colnames
        assert f"{prefix_clf}_goodness_of_fit" in events.colnames
        assert f"{prefix_disp}_alt" in events.colnames
        assert f"{prefix_disp}_az" in events.colnames
        assert f"{prefix_disp}_ang_distance_uncert" in events.colnames
        assert f"{prefix_disp}_is_valid" in events.colnames
        assert f"{prefix_disp}_goodness_of_fit" in events.colnames

        tel_events = loader.read_telescope_events()
        assert f"{prefix_clf}_prediction" in tel_events.colnames
        assert f"{prefix_clf}_telescopes" in tel_events.colnames
        assert f"{prefix_clf}_is_valid" in tel_events.colnames
        assert f"{prefix_clf}_goodness_of_fit" in tel_events.colnames
        assert f"{prefix_disp}_alt" in tel_events.colnames
        assert f"{prefix_disp}_az" in tel_events.colnames
        assert f"{prefix_disp}_ang_distance_uncert" in tel_events.colnames
        assert f"{prefix_disp}_is_valid" in tel_events.colnames
        assert f"{prefix_disp}_goodness_of_fit" in tel_events.colnames

        assert f"{prefix_clf}_tel_prediction" in tel_events.colnames
        assert f"{prefix_clf}_tel_is_valid" in tel_events.colnames
        assert f"{prefix_disp}_tel_alt" in tel_events.colnames
        assert f"{prefix_disp}_tel_az" in tel_events.colnames
        assert f"{prefix_disp}_tel_is_valid" in tel_events.colnames
        assert f"{prefix_disp}_parameter_norm" in tel_events.colnames
        assert f"{prefix_disp}_parameter_is_valid" in tel_events.colnames

        # check that the "--no-dl1-parameters" option worked
        assert "hillas_intensity" not in tel_events.colnames
        assert "ExtraTreesRegressor_energy" in events.colnames

    trigger = read_table(output_path, "/dl1/event/subarray/trigger")
    particle_clf = read_table(
        output_path, f"/dl2/event/subarray/classification/{prefix_clf}"
    )
    check_equal_array_event_order(trigger, particle_clf)
    disp_reco = read_table(output_path, f"/dl2/event/subarray/geometry/{prefix_disp}")
    check_equal_array_event_order(trigger, disp_reco)
