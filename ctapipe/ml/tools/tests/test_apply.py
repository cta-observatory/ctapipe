from ctapipe.core import run_tool
from ctapipe.io import TableLoader
import shutil


def test_apply_energy_regressor(energy_regressor_path, dl1_parameters_file, tmp_path):
    from ctapipe.ml.tools.apply_energy_regressor import ApplyEnergyRegressor

    input_path = tmp_path / dl1_parameters_file.name

    # create copy to not mutate common test file
    shutil.copy2(dl1_parameters_file, input_path)

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
