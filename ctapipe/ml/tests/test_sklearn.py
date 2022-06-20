import numpy as np
import pytest
from astropy.table import Table
import astropy.units as u
from ctapipe.core import Component
from ctapipe.ml.sklearn import Classifier, Regressor
from numpy.testing import assert_array_equal
from traitlets import TraitError
from traitlets.config import Config


def test_supported_regressors():
    from ctapipe.ml.sklearn import SUPPORTED_REGRESSORS
    from sklearn.ensemble import RandomForestRegressor

    assert "RandomForestRegressor" in SUPPORTED_REGRESSORS
    assert SUPPORTED_REGRESSORS["RandomForestRegressor"] is RandomForestRegressor


def test_supported_classifiers():
    from ctapipe.ml.sklearn import SUPPORTED_CLASSIFIERS
    from sklearn.ensemble import RandomForestClassifier

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
    t["energy"] = y * u.TeV
    t["X0"][10] = np.nan
    t["X1"][30] = np.nan

    X, y = make_blobs(n_samples=100, n_features=3, centers=2, random_state=0)
    for i, col in enumerate(X.T, start=5):
        t[f"X{i}"] = col
    t["particle"] = y

    return t


def test_model_init():
    from ctapipe.ml.sklearn import Classifier
    from sklearn.ensemble import RandomForestClassifier

    # need to provide a model_cls
    with pytest.raises(TraitError):
        Classifier()

    # cannot be a regressor
    with pytest.raises(TraitError):
        Classifier(model_cls="RandomForestRegressor")

    # should create class with sklearn defaults
    c = Classifier(model_cls="RandomForestClassifier")
    assert isinstance(c.model, RandomForestClassifier)

    config = Config(
        {
            "Classifier": {
                "model_cls": "RandomForestClassifier",
                "model_config": {"n_estimators": 10, "max_depth": 15},
            }
        }
    )

    # should create class with sklearn defaults
    c = Classifier(config=config)
    assert isinstance(c.model, RandomForestClassifier)
    assert c.model.n_estimators == 10
    assert c.model.max_depth == 15


@pytest.mark.parametrize("model_cls", ["LinearRegression", "RandomForestRegressor"])
def test_regressor(model_cls, example_table):
    from ctapipe.ml.sklearn import Regressor

    regressor = Regressor(
        model_cls=model_cls, target="energy", features=[f"X{i}" for i in range(8)]
    )

    regressor.fit(example_table)
    prediction, valid = regressor.predict(example_table)
    assert prediction.shape == (100,)
    assert not valid[10]
    assert not valid[30]
    assert np.isfinite(prediction[valid]).all()
    assert np.isnan(prediction[~valid]).all()


@pytest.mark.parametrize("model_cls", ["LinearRegression", "RandomForestRegressor"])
def test_regressor_single_event(model_cls, example_table):
    from ctapipe.ml.sklearn import Regressor

    regressor = Regressor(
        model_cls=model_cls, target="energy", features=[f"X{i}" for i in range(8)]
    )

    regressor.fit(example_table)
    prediction, valid = regressor.predict(example_table[[0]])
    assert prediction.unit == u.TeV
    assert prediction.shape == (1,)

    # now test with a single invalid event
    invalid = example_table[[0]].copy()
    for col in filter(lambda col: col.startswith("X"), invalid.colnames):
        invalid[col][:] = np.nan

    prediction, valid = regressor.predict(invalid)
    assert prediction.shape == (1,)
    assert valid[0] == False


def test_regressor_log_target(example_table):
    from ctapipe.ml.sklearn import Regressor

    regressor = Regressor(
        model_cls="LinearRegression",
        target="energy",
        log_target=True,
        features=[f"X{i}" for i in range(8)],
    )

    regressor.fit(example_table)
    prediction, valid = regressor.predict(example_table)
    assert prediction.shape == (100,)
    assert np.isnan(prediction[10])
    assert np.isnan(prediction[30])
    assert not valid[10]
    assert not valid[30]


@pytest.mark.parametrize(
    "model_cls", ["KNeighborsClassifier", "RandomForestClassifier"]
)
def test_classifier(model_cls, example_table):
    from ctapipe.ml.sklearn import Classifier

    classifier = Classifier(
        model_cls=model_cls, target="particle", features=[f"X{i}" for i in range(8)]
    )

    classifier.fit(example_table)
    prediction, valid = classifier.predict(example_table)
    assert prediction.shape == (100,)
    assert_array_equal(np.unique(prediction), [-1, 0, 1])
    assert prediction[10] == -1
    assert prediction[30] == -1
    assert not valid[10]
    assert not valid[30]

    score, valid = classifier.predict_score(example_table)
    assert score.shape == (100,)
    assert np.isnan(score[10])
    assert np.isnan(score[30])
    assert not valid[10]
    assert not valid[30]

    valid = np.isfinite(score)
    assert_array_equal((score[valid] > 0.5).astype(int), prediction[valid])


def test_io(example_table, tmp_path):
    from ctapipe.ml.sklearn import Classifier, Regressor

    classifier = Classifier(
        model_cls="RandomForestClassifier",
        model_config=dict(n_estimators=5, max_depth=3),
        target="particle",
        features=[f"X{i}" for i in range(8)],
    )

    classifier.fit(example_table)
    path = tmp_path / "classifier.pkl"

    classifier.write(path)
    loaded = Classifier.load(path)
    assert loaded.features == classifier.features
    assert_array_equal(
        loaded.model.feature_importances_, classifier.model.feature_importances_
    )

    with pytest.raises(TypeError):
        Regressor.load(path)


class Parent(Component):
    def __init__(self, config):
        super().__init__(config=config)
        self.classifier = Classifier(parent=self)


def test_io_with_parent(example_table, tmp_path):
    config = Config(
        dict(
            Classifier=dict(
                model_cls="RandomForestClassifier",
                model_config=dict(n_estimators=5, max_depth=3),
                target="particle",
                features=[f"X{i}" for i in range(8)],
            )
        )
    )

    parent = Parent(config=config)
    parent.classifier.fit(example_table)
    path = tmp_path / "classifier.pkl"

    parent.classifier.write(path)
    loaded = Classifier.load(path)
    assert loaded.features == parent.classifier.features
    assert_array_equal(
        loaded.model.feature_importances_, parent.classifier.model.feature_importances_
    )

    with pytest.raises(TypeError):
        Regressor.load(path)
