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
def test_regressor(model_cls):
    from sklearn.datasets import make_regression
    from ctapipe.ml.sklearn import Regressor

    X, y = make_regression(n_samples=100, n_features=5, n_informative=3, random_state=0)

    t = Table({f"X{i}": col for i, col in enumerate(X.T)})
    t["target"] = y
    t["X0"][10] = np.nan
    t["X1"][30] = np.nan

    regressor = Regressor(
        model_cls=model_cls, target="target", features=[f"X{i}" for i in range(5)]
    )

    regressor.fit(t)
    prediction = regressor.predict(t)
    assert prediction.shape == (100,)
    assert np.isnan(prediction[10])
    assert np.isnan(prediction[30])


@pytest.mark.parametrize(
    "model_cls", ["KNeighborsClassifier", "RandomForestClassifier"]
)
def test_classifier(model_cls):
    from sklearn.datasets import make_blobs
    from ctapipe.ml.sklearn import Classifier

    X, y = make_blobs(n_samples=100, n_features=3, centers=2, random_state=0)

    t = Table({f"X{i}": col for i, col in enumerate(X.T)})
    t["target"] = y
    t["X0"][10] = np.nan
    t["X1"][30] = np.nan

    classifier = Classifier(
        model_cls=model_cls,
        target="target",
        features=[f"X{i}" for i in range(X.shape[1])],
    )

    classifier.fit(t)
    prediction = classifier.predict(t)
    assert prediction.shape == (100,)
    assert_array_equal(np.unique(prediction), [-1, 0, 1])
    assert prediction[10] == -1
    assert prediction[30] == -1

    score = classifier.predict_score(t)
    assert score.shape == (100,)
    assert np.isnan(score[10])
    assert np.isnan(score[30])

    valid = np.isfinite(score)
    assert_array_equal((score[valid] > 0.5).astype(int), prediction[valid])
