def test_train_energy_regressor(energy_regressor_path):
    from ctapipe.ml.sklearn import Regressor
    Regressor.load(energy_regressor_path)
