def test_train_energy_regressor(energy_regressor_path):
    from ctapipe.ml.sklearn import Regressor

    Regressor.load(energy_regressor_path)


def test_train_particle_classifier(particle_classifier_path):
    from ctapipe.ml.sklearn import Classifier

    Classifier.load(particle_classifier_path)
