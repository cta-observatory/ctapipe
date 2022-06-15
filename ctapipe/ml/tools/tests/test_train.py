from ctapipe.core import run_tool
from ctapipe.utils import resource_file


def test_train_energy_regressor(tmp_path):
    from ctapipe.ml import TrainEnergyRegressor
    from ctapipe.ml.sklearn import Regressor

    tool = TrainEnergyRegressor()
    config = resource_file("ml-config.yaml")
    out_file = tmp_path / "test_train_energy_regressor.pkl"
    ret = run_tool(
        tool,
        argv=[
            "--input=dataset://gamma_diffuse_dl2_train_small.dl2.h5",
            f"--output={out_file}",
            f"--config={config}",
            "--log-level=INFO",
        ],
    )
    assert ret == 0
    Regressor.load(out_file)
