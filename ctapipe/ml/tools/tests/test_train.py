from ctapipe.core import run_tool
from ctapipe.ml import TrainEnergyRegressor
from ctapipe.utils import resource_file


def test_train_energy_regressor(tmp_path, dl2_shower_geometry_file):
    tool = TrainEnergyRegressor()
    tool.load_config_file(resource_file("ml-config.yaml"))
    out_file = tmp_path / "test_train_energy_regressor.pkl"
    assert (
        run_tool(
            tool,
            argv=[
                f"--input={dl2_shower_geometry_file}",
                f"--output={out_file}",
                "--TrainEnergyRegressor.n_cross_validation=2",  # default of 5 does not work with sample size of 2
            ],
        )
        == 0
    )
