import pytest

from ctapipe.core import run_tool
from ctapipe.exceptions import TooFewEvents
from ctapipe.utils.datasets import resource_file


def test_train_energy_regressor(energy_regressor_path):
    from ctapipe.reco import EnergyRegressor

    EnergyRegressor.read(energy_regressor_path)


def test_train_particle_classifier(particle_classifier_path):
    from ctapipe.reco import ParticleClassifier

    ParticleClassifier.read(particle_classifier_path)


def test_train_disp_reconstructor(disp_reconstructor_path):
    from ctapipe.reco import DispReconstructor

    DispReconstructor.read(disp_reconstructor_path)


def test_too_few_events(tmp_path, dl2_shower_geometry_file):
    from ctapipe.tools.train_energy_regressor import TrainEnergyRegressor

    tool = TrainEnergyRegressor()
    config = resource_file("train_energy_regressor.yaml")
    out_file = tmp_path / "energy.pkl"

    with pytest.raises(TooFewEvents, match="No events after quality query"):
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


def test_sampling(tmp_path, dl2_shower_geometry_file):
    from ctapipe.tools.train_energy_regressor import TrainEnergyRegressor

    tool = TrainEnergyRegressor()
    config = resource_file("train_energy_regressor.yaml")
    out_file = tmp_path / "energy.pkl"

    run_tool(
        tool,
        argv=[
            "--input=dataset://gamma_diffuse_dl2_train_small.dl2.h5",
            f"--output={out_file}",
            f"--config={config}",
            "--log-level=INFO",
            "--n-events=100",
        ],
        raises=True,
    )


def test_cross_validation_results(tmp_path, gamma_train_clf, proton_train_clf):
    from ctapipe.tools.train_disp_reconstructor import TrainDispReconstructor
    from ctapipe.tools.train_energy_regressor import TrainEnergyRegressor
    from ctapipe.tools.train_particle_classifier import TrainParticleClassifier

    tool = TrainEnergyRegressor()
    config = resource_file("train_energy_regressor.yaml")
    out_file = tmp_path / "energy_.pkl"
    energy_cv_out_file = tmp_path / "energy_cv_results.h5"

    ret = run_tool(
        tool,
        argv=[
            "--input=dataset://gamma_diffuse_dl2_train_small.dl2.h5",
            f"--output={out_file}",
            f"--config={config}",
            f"--cv-output={energy_cv_out_file}",
            "--log-level=INFO",
        ],
    )
    assert ret == 0
    assert energy_cv_out_file.exists()

    tool = TrainParticleClassifier()
    config = resource_file("train_particle_classifier.yaml")
    out_file = tmp_path / "particle_classifier_.pkl"
    classifier_cv_out_file = tmp_path / "classifier_cv_results.h5"

    ret = run_tool(
        tool,
        argv=[
            f"--signal={gamma_train_clf}",
            f"--background={proton_train_clf}",
            f"--output={out_file}",
            f"--config={config}",
            f"--cv-output={classifier_cv_out_file}",
            "--log-level=INFO",
        ],
    )
    assert ret == 0
    assert classifier_cv_out_file.exists()

    tool = TrainDispReconstructor()
    config = resource_file("train_disp_reconstructor.yaml")
    out_file = tmp_path / "disp_reconstructor_.pkl"
    disp_cv_out_file = tmp_path / "disp_cv_results.h5"

    ret = run_tool(
        tool,
        argv=[
            f"--input={gamma_train_clf}",
            f"--output={out_file}",
            f"--config={config}",
            f"--cv-output={disp_cv_out_file}",
            "--log-level=INFO",
        ],
    )
    assert ret == 0
    assert disp_cv_out_file.exists()


def test_no_cross_validation(tmp_path):
    from ctapipe.tools.train_energy_regressor import TrainEnergyRegressor

    out_file = tmp_path / "energy.pkl"

    tool = TrainEnergyRegressor()
    config = resource_file("train_energy_regressor.yaml")
    ret = run_tool(
        tool,
        argv=[
            "--input=dataset://gamma_diffuse_dl2_train_small.dl2.h5",
            f"--output={out_file}",
            f"--config={config}",
            "--CrossValidator.n_cross_validations=0",
            "--log-level=INFO",
            "--overwrite",
        ],
    )
    assert ret == 0
    return out_file
