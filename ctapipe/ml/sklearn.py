"""
Component Wrappers around sklearn models
"""

from copy import deepcopy
import numpy as np
from traitlets import Dict, List, Unicode, Enum, Integer, Bool
import joblib
from sklearn.utils import all_estimators

from ..core import Component
from .preprocessing import check_valid_rows, table_to_float


SUPPORTED_CLASSIFIERS = dict(all_estimators("classifier"))
SUPPORTED_REGRESSORS = dict(all_estimators("regressor"))
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

    def table_to_y(self, table, mask=None):
        return np.array(table[self.target][mask])

    def fit(self, table):
        X, valid = self.table_to_X(table)
        y = self.table_to_y(table, mask=valid)
        self.model.fit(X, y)

    def predict(self, table):
        X, valid = self.table_to_X(table)
        n_outputs = getattr(self.model, "n_outputs_", 1)

        if n_outputs > 1:
            shape = (len(table), n_outputs)
        else:
            shape = (len(table),)

        prediction = np.full(shape, np.nan)
        if np.any(valid):
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

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_trait_values']['parent'] = None
        state['_trait_notifiers'] = {}
        return state


class Regressor(Model):
    model_cls = Enum(
        SUPPORTED_REGRESSORS.keys(), default_value=None, allow_none=False
    ).tag(config=True)

    log_target = Bool(default_value=False).tag(config=True)

    def table_to_y(self, table, mask=None):
        y = super().table_to_y(table, mask=mask)

        if self.log_target:
            if np.any(y <= 0):
                raise ValueError("y contains negative values, cannot apply log")
            return np.log(y)
        return y

    def predict(self, table):
        prediction = super().predict(table)

        if self.log_target:
            with np.errstate(invalid="ignore"):
                return np.exp(prediction)

        return prediction


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
        if np.any(valid):
            prediction[valid] = self.model.predict(X)
        return prediction

    def predict_score(self, table):
        X, valid = self.table_to_X(table)

        n_classes = getattr(self.model, "n_classes_", 2)
        n_rows = len(table)
        shape = (n_rows, n_classes) if n_classes > 2 else (n_rows,)

        scores = np.full(shape, np.nan)

        if np.any(valid):
            prediction = self.model.predict_proba(X)[:]

            if n_classes > 2:
                scores[valid] = prediction
            else:
                # only return one score for the positive class
                scores[valid] = prediction[:, 1]

        return scores
