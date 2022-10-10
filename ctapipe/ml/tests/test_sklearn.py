import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table
from numpy.testing import assert_array_equal
from traitlets import TraitError
from traitlets.config import Config

from ctapipe.core import Component
from ctapipe.ml.sklearn import EnergyRegressor, ParticleIdClassifier

KEY = "LST_LST_LSTCam"


def test_supported_regressors():
    from sklearn.ensemble import RandomForestRegressor

    from ctapipe.ml.sklearn import SUPPORTED_REGRESSORS

    assert "RandomForestRegressor" in SUPPORTED_REGRESSORS
    assert SUPPORTED_REGRESSORS["RandomForestRegressor"] is RandomForestRegressor


def test_supported_classifiers():
    from sklearn.ensemble import RandomForestClassifier

    from ctapipe.ml.sklearn import SUPPORTED_CLASSIFIERS

    assert "RandomForestClassifier" in SUPPORTED_CLASSIFIERS
    assert SUPPORTED_CLASSIFIERS["RandomForestClassifier"] is RandomForestClassifier


def make_positive_regression(n_samples, n_features, n_informative, random_state=0):
    rng = np.random.default_rng(random_state)

    y = rng.uniform(1, 1000, n_samples)

    X = np.zeros((n_samples, n_features))

    coeffs = rng.normal(0, 10, size=n_informative)
    X[:, :n_informative] = y[:, np.newaxis] * coeffs
    # add some noise
    noise = rng.normal(0, np.sqrt(y)[:, np.newaxis], (n_samples, n_informative))
    X[:, :n_informative] += noise

    n_random = n_features - n_informative
    means = rng.uniform(-10, 10, n_random)[np.newaxis, :]
    stds = rng.uniform(0.1, 10, n_random)[np.newaxis, :]
    X[:, n_informative:] = rng.normal(means, stds, (n_samples, n_random))
    return X, y


@pytest.fixture()
def example_table():
    from sklearn.datasets import make_blobs

    X, y = make_positive_regression(
        n_samples=100, n_features=5, n_informative=3, random_state=0
    )
    t = Table({f"X{i}": col for i, col in enumerate(X.T)})
    t["true_energy"] = y * u.TeV
    t["X0"][10] = np.nan
    t["X1"][30] = np.nan

    X, y = make_blobs(n_samples=100, n_features=3, centers=2, random_state=0)
    for i, col in enumerate(X.T, start=5):
        t[f"X{i}"] = col
    t["true_shower_primary_id"] = y

    return t


def test_model_init(example_subarray):
    from sklearn.ensemble import RandomForestClassifier

    # need to provide a model_cls
    with pytest.raises(TraitError):
        ParticleIdClassifier(example_subarray)

    # cannot be a regressor
    with pytest.raises(TraitError):
        ParticleIdClassifier(example_subarray, model_cls="RandomForestRegressor")

    # should create class with sklearn defaults
    c = ParticleIdClassifier(example_subarray, model_cls="RandomForestClassifier")
    assert isinstance(c._new_model(), RandomForestClassifier)

    config = Config(
        {
            "ParticleIdClassifier": {
                "model_cls": "RandomForestClassifier",
                "model_config": {"n_estimators": 20, "max_depth": 15},
            }
        }
    )

    c = ParticleIdClassifier(example_subarray, config=config)
    clf = c._new_model()
    assert isinstance(clf, RandomForestClassifier)
    assert clf.n_estimators == 20
    assert clf.max_depth == 15


@pytest.mark.parametrize("model_cls", ["LinearRegression", "RandomForestRegressor"])
@pytest.mark.parametrize("log_target", (False, True))
def test_regressor(model_cls, example_table, log_target, example_subarray):
    regressor = EnergyRegressor(
        example_subarray,
        model_cls=model_cls,
        features=[f"X{i}" for i in range(8)],
        log_target=log_target,
    )

    regressor.fit(KEY, example_table)
    table = regressor.predict_table(KEY, example_table)
    prediction = table[f"{model_cls}_tel_energy"].quantity
    valid = table[f"{model_cls}_tel_is_valid"]
    assert prediction.shape == (100,)
    assert prediction.unit == u.TeV
    assert not valid[10]
    assert not valid[30]
    assert np.isfinite(prediction[valid]).all()
    assert np.isnan(prediction[~valid]).all()

    assert regressor.stereo_combiner.property == "energy"
    assert regressor.stereo_combiner.prefix == model_cls


@pytest.mark.parametrize("model_cls", ["LinearRegression", "RandomForestRegressor"])
def test_regressor_single_event(model_cls, example_table, example_subarray):
    regressor = EnergyRegressor(
        example_subarray, model_cls=model_cls, features=[f"X{i}" for i in range(8)]
    )
    regressor.fit(KEY, example_table)

    table = regressor.predict_table(KEY, example_table[[0]])
    prediction = table[f"{model_cls}_tel_energy"].quantity
    valid = table[f"{model_cls}_tel_is_valid"]
    assert prediction.unit == u.TeV
    assert prediction.shape == (1,)

    # now test with a single invalid event
    invalid = example_table[[0]].copy()
    for col in filter(lambda col: col.startswith("X"), invalid.colnames):
        invalid[col][:] = np.nan

    table = regressor.predict_table(KEY, invalid)
    prediction = table[f"{model_cls}_tel_energy"].quantity
    valid = table[f"{model_cls}_tel_is_valid"]
    assert prediction.shape == (1,)
    assert valid[0] == False


@pytest.mark.parametrize(
    "model_cls", ["KNeighborsClassifier", "RandomForestClassifier"]
)
def test_classifier(model_cls, example_table, example_subarray):
    classifier = ParticleIdClassifier(
        example_subarray,
        model_cls=model_cls,
        features=[f"X{i}" for i in range(8)],
    )

    classifier.fit(KEY, example_table)
    prediction, valid = classifier._predict(KEY, example_table)
    assert prediction.shape == (100,)
    assert_array_equal(np.unique(prediction), [-1, 0, 1])
    assert prediction[10] == -1
    assert prediction[30] == -1
    assert not valid[10]
    assert not valid[30]

    score, valid = classifier._predict_score(KEY, example_table)
    assert score.shape == (100,)
    assert np.isnan(score[10])
    assert np.isnan(score[30])
    assert not valid[10]
    assert not valid[30]

    valid = np.isfinite(score)
    assert_array_equal((score[valid] < 0.5).astype(int), prediction[valid])

    result_table = classifier.predict_table(KEY, example_table)
    score = result_table[f"{model_cls}_tel_prediction"].quantity
    valid = result_table[f"{model_cls}_tel_is_valid"]
    assert score.shape == (100,)
    assert np.isnan(score[10])
    assert np.isnan(score[30])
    assert not valid[10]
    assert not valid[30]

    valid = np.isfinite(score)
    assert_array_equal((score[valid] < 0.5).astype(int), prediction[valid])


def test_io_with_parent(example_table, tmp_path, example_subarray):
    class Parent(Component):
        def __init__(self, config):
            super().__init__(config=config)
            self.classifier = ParticleIdClassifier(
                parent=self,
                subarray=example_subarray,
            )

    config = Config(
        dict(
            ParticleIdClassifier=dict(
                model_cls="RandomForestClassifier",
                model_config=dict(n_estimators=5, max_depth=3),
                features=[f"X{i}" for i in range(8)],
            )
        )
    )

    parent = Parent(config=config)
    parent.classifier.fit(KEY, example_table)
    path = tmp_path / "classifier.pkl"

    parent.classifier.write(path)
    loaded = ParticleIdClassifier(load_path=path)
    assert loaded.features == parent.classifier.features
    assert_array_equal(
        loaded._models[KEY].feature_importances_,
        parent.classifier._models[KEY].feature_importances_,
    )

    with pytest.raises(TypeError):
        EnergyRegressor.read(path)
