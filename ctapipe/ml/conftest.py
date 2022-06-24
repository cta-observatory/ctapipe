import pytest
from ctapipe.utils.datasets import resource_file
from ctapipe.core import run_tool

@pytest.fixture(scope='session')
def model_tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("models")


@pytest.fixture(scope='session')
def energy_regressor_path(model_tmp_path):
    from ctapipe.ml.tools.train_energy_regressor import TrainEnergyRegressor

    tool = TrainEnergyRegressor()
    config = resource_file("ml-config.yaml")
    out_file = model_tmp_path / "energy.pkl"
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
    return out_file
