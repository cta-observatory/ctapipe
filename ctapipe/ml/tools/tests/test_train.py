def test_train_energy_regressor(energy_regressor_path):
    from ctapipe.ml import EnergyRegressor
    EnergyRegressor.read(energy_regressor_path)


def test_train_particle_classifier(particle_classifier_path):
    from ctapipe.ml import ParticleIdClassifier
    ParticleIdClassifier.read(particle_classifier_path)
