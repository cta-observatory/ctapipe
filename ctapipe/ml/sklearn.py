"""
Component Wrappers around sklearn models
"""
from inspect import isabstract
from importlib import import_module

import numpy as np
from traitlets import Dict, List, Unicode, Enum, Integer
from sklearn.base import is_classifier, is_regressor
import joblib

from ..core import Component
from .preprocessing import check_valid_rows, table_to_float


sklearn_modules = {
    name: import_module(f"sklearn.{name}")
    for name in ["ensemble", "linear_model", "neighbors", "svm", "tree", "naive_bayes"]
}


def get_models(filter_func=is_classifier):
    models = {}
    for module in sklearn_modules.values():
        for cls_name in dir(module):
            cls = getattr(module, cls_name)
            if filter_func(cls) and not isabstract(cls):
                models[cls.__name__] = cls

    return models


SUPPORTED_CLASSIFIERS = get_models(is_classifier)
SUPPORTED_REGRESSORS = get_models(is_regressor)
SUPPORTED_MODELS = {**SUPPORTED_CLASSIFIERS, **SUPPORTED_REGRESSORS}


class Model(Component):
    features = List(Unicode(), help="Features to use for this model").tag(config=True)
    model_config = Dict({}, help="kwargs for the sklearn model").tag(config=True)
    model_cls = Enum(SUPPORTED_MODELS.keys(), default_value=None, allow_none=False).tag(
        config=True
    )
    target = Unicode(None, allow_none=False, help="Name of target column").tag(
        config=True
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = SUPPORTED_MODELS[self.model_cls](**self.model_config)

    def table_to_X(self, table):
        feature_table = table[self.features]
        valid = check_valid_rows(feature_table, log=self.log)
        X = table_to_float(feature_table[valid])
        return X, valid

    def table_to_y(self, table):
        return np.array(table[self.target])

    def fit(self, table):
        X, valid = self.table_to_X(table)
        y = self.table_to_y(table)[valid]
        self.model.fit(X, y)

    def predict(self, table):
        X, valid = self.table_to_X(table)
        n_outputs = getattr(self.model, "n_outputs_", 1)

        if n_outputs > 1:
            shape = (len(table), n_outputs)
        else:
            shape = (len(table),)

        prediction = np.full(shape, np.nan)
        prediction[valid] = self.model.predict(X)
        return prediction

    def write(self, path):
        with open(path, "wb") as f:
            joblib.dump(self, f, compress=True)

    @classmethod
    def load(cls, path, check_cls=True):
        with open(path, "rb") as f:
            model = joblib.load(f)

        if check_cls is True and not model.__class__ is cls:
            raise TypeError(
                "File did not contain an instance of {cls}, got {model.__class__}"
            )

        return model


class Regressor(Model):
    model_cls = Enum(
        SUPPORTED_REGRESSORS.keys(), default_value=None, allow_none=False
    ).tag(config=True)


class Classifier(Model):
    model_cls = Enum(
        SUPPORTED_CLASSIFIERS.keys(), default_value=None, allow_none=False
    ).tag(config=True)
    invalid_class = Integer(-1).tag(config=True)

    def predict(self, table):
        X, valid = self.table_to_X(table)
        n_outputs = getattr(self.model, "n_outputs_", 1)

        if n_outputs > 1:
            shape = (len(table), n_outputs)
        else:
            shape = (len(table),)

        prediction = np.full(shape, self.invalid_class, dtype=np.int8)
        prediction[valid] = self.model.predict(X)
        return prediction

    def predict_score(self, table):
        X, valid = self.table_to_X(table)

        n_classes = getattr(self.model, "n_classes_", 2)
        n_rows = len(table)
        shape = (n_rows, n_classes) if n_classes > 2 else (n_rows,)

        scores = np.full(shape, np.nan)
        prediction = self.model.predict_proba(X)[:]

        if n_classes > 2:
            scores[valid] = prediction
        else:
            # only return one score for the positive class
            scores[valid] = prediction[:, 1]

        return scores
