"""
Test fixtures for the ml module
"""
import pytest

from ctapipe.core import run_tool
from ctapipe.utils.datasets import resource_file
from ctapipe.utils.filelock import FileLock


@pytest.fixture(scope="session")
def model_tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("models")


@pytest.fixture(scope="session")
def energy_regressor_path(model_tmp_path):
    from ctapipe.ml.tools.train_energy_regressor import TrainEnergyRegressor

    out_file = model_tmp_path / "energy.pkl"

    with FileLock(out_file.with_suffix(out_file.suffix + ".lock")):
        if out_file.is_file():
            return out_file

        tool = TrainEnergyRegressor()
        config = resource_file("ml_config.yaml")
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


@pytest.fixture(scope="session")
def particle_classifier_path(model_tmp_path):
    from ctapipe.ml.tools.train_particle_classifier import TrainParticleIdClassifier

    out_file = model_tmp_path / "particle_classifier.pkl"
    with FileLock(out_file.with_suffix(out_file.suffix + ".lock")):
        if out_file.is_file():
            return out_file

        tool = TrainParticleIdClassifier()
        config = resource_file("ml_config.yaml")

        ret = run_tool(
            tool,
            argv=[
                "--signal=dataset://gamma_diffuse_dl2_train_small.dl2.h5",
                "--background=dataset://proton_dl2_train_small.dl2.h5",
                f"--output={out_file}",
                f"--config={config}",
                "--log-level=INFO",
            ],
        )
        assert ret == 0
        return out_file


@pytest.fixture(scope="session")
def disp_reconstructor_paths(model_tmp_path):
    from ctapipe.ml.tools.train_disp_reconstructor import TrainDispReconstructor

    tool = TrainDispReconstructor()
    config = resource_file("ml_config.yaml")
    out_file_reg = model_tmp_path / "disp_regressor.pkl"
    out_file_clf = model_tmp_path / "sign_classifier.pkl"
    ret = run_tool(
        tool,
        argv=[
            "--input=dataset://gamma_diffuse_dl2_train_small.dl2.h5",
            f"--output-regressor={out_file_reg}",
            f"--output-classifier={out_file_clf}",
            f"--config={config}",
            "--log-level=INFO",
        ],
    )
    assert ret == 0
    return out_file_reg, out_file_clf