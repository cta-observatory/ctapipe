import pytest

import numpy as np
from traitlets.config import Config
from traitlets import TraitError
from astropy.table import Table
from numpy.testing import assert_array_equal


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


@pytest.fixture()
def example_table():
    from sklearn.datasets import make_regression
    from sklearn.datasets import make_blobs

    X, y = make_regression(n_samples=100, n_features=5, n_informative=3, random_state=0)
    t = Table({f"X{i}": col for i, col in enumerate(X.T)})
    t["energy"] = y
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
    prediction = regressor.predict(example_table)
    assert prediction.shape == (100,)
    assert np.isnan(prediction[10])
    assert np.isnan(prediction[30])


@pytest.mark.parametrize(
    "model_cls", ["KNeighborsClassifier", "RandomForestClassifier"]
)
def test_classifier(model_cls, example_table):
    from ctapipe.ml.sklearn import Classifier

    classifier = Classifier(
        model_cls=model_cls, target="particle", features=[f"X{i}" for i in range(8)]
    )

    classifier.fit(example_table)
    prediction = classifier.predict(example_table)
    assert prediction.shape == (100,)
    assert_array_equal(np.unique(prediction), [-1, 0, 1])
    assert prediction[10] == -1
    assert prediction[30] == -1

    score = classifier.predict_score(example_table)
    assert score.shape == (100,)
    assert np.isnan(score[10])
    assert np.isnan(score[30])

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
