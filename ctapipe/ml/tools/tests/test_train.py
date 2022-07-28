import pytest

from ctapipe.core import run_tool
from ctapipe.utils.datasets import resource_file


def test_train_energy_regressor(energy_regressor_path):
    from ctapipe.ml import EnergyRegressor

    EnergyRegressor.read(energy_regressor_path)


def test_train_particle_classifier(particle_classifier_path):
    from ctapipe.ml import ParticleIdClassifier

    ParticleIdClassifier.read(particle_classifier_path)


def test_too_few_events(
    caplog, model_tmp_path, dl2_shower_geometry_file, dl2_proton_geometry_file
):
    from ctapipe.ml.tools.train_energy_regressor import TrainEnergyRegressor
    from ctapipe.ml.tools.train_particle_classifier import TrainParticleIdClassifier

    tool = TrainEnergyRegressor()
    config = resource_file("ml_config.yaml")
    out_file = model_tmp_path / "energy.pkl"

    with pytest.raises(ValueError):
        run_tool(
            tool,
            argv=[
                f"--input={dl2_shower_geometry_file}",
                f"--output={out_file}",
                f"--config={config}",
                "--log-level=INFO",
            ],
            raises=True,
        )

    tool = TrainParticleIdClassifier()
    out_file = model_tmp_path / "particle_classifier.pkl"

    with pytest.raises(ValueError):
        run_tool(
            tool,
            argv=[
                f"--signal={dl2_shower_geometry_file}",
                f"--background={dl2_proton_geometry_file}",
                f"--output={out_file}",
                f"--config={config}",
                "--log-level=INFO",
            ],
            raises=True,
        )
