"""
Component Wrappers around sklearn models
"""
from abc import abstractmethod
from collections import defaultdict

import astropy.units as u
import joblib
import numpy as np
from astropy.table import QTable, Table, vstack
from astropy.utils.decorators import lazyproperty
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils import all_estimators
from tqdm import tqdm

from ..containers import (
    ArrayEventContainer,
    ParticleClassificationContainer,
    ReconstructedEnergyContainer,
)
from ..core import Component, FeatureGenerator, Provenance, QualityQuery
from ..core.traits import (
    Bool,
    Dict,
    Enum,
    Int,
    Integer,
    List,
    Path,
    TraitError,
    Unicode,
    create_class_enum_trait,
)
from ..io import write_table
from ..reco import Reconstructor
from .preprocessing import check_valid_rows, collect_features, table_to_float
from .stereo_combination import StereoCombiner
from .utils import add_defaults_and_meta

__all__ = [
    "SKLearnReconstructor",
    "SKLearnRegressionReconstructor",
    "SKLearnClassficationReconstructor",
    "EnergyRegressor",
    "ParticleIdClassifier",
]


SUPPORTED_CLASSIFIERS = dict(all_estimators("classifier"))
SUPPORTED_REGRESSORS = dict(all_estimators("regressor"))
SUPPORTED_MODELS = {**SUPPORTED_CLASSIFIERS, **SUPPORTED_REGRESSORS}


class SKLearnReconstructor(Reconstructor):
    """Base Class for a Machine Learning Based Reconstructor.

    Keeps a dictionary of sklearn models, the current tools are designed
    to train one model per telescope type.

    Models can either be instantiated normally, or they can load themselves
    from a pickled file if the ``load_path`` traitlet is given.
    """

    #: What particle property is being predicted, one of "energy", "direction", "classification"
    property = None
    #: Name of the target column in training table
    target = None

    features = List(Unicode(), help="Features to use for this model").tag(config=True)
    model_config = Dict({}, help="kwargs for the sklearn model").tag(config=True)
    model_cls = Enum(SUPPORTED_MODELS.keys(), default_value=None, allow_none=True).tag(
        config=True
    )

    stereo_combiner_cls = create_class_enum_trait(
        StereoCombiner,
        default_value="StereoMeanCombiner",
        help="Which stereo combination method to use",
    ).tag(config=True)

    load_path = Path(
        default_value=None,
        allow_none=True,
        help="If given, load serialized model from this path",
    ).tag(config=True)

    def __init__(self, subarray=None, models=None, **kwargs):
        # Run the Component __init__ first to handle the configuration
        # and make `self.load_path` available
        Component.__init__(self, **kwargs)

        if self.load_path is None:
            if self.model_cls is None:
                raise TraitError(
                    "Must provide `model_cls` if not loading model from file"
                )

            if subarray is None:
                raise TypeError(
                    "__init__() missing 1 required positional argument: 'subarray'"
                )

            super().__init__(subarray, **kwargs)
            self.subarray = subarray
            self.qualityquery = QualityQuery(parent=self)
            self.generate_features = FeatureGenerator(parent=self)

            # to verify settings
            self._new_model()

            self._models = {} if models is None else models
            self.unit = None
            self.stereo_combiner = StereoCombiner.from_name(
                self.stereo_combiner_cls,
                prefix=self.model_cls,
                property=self.property,
                parent=self,
            )
        else:
            loaded = self.read(self.load_path)
            if (
                subarray is not None
                and loaded.subarray.telescope_types != subarray.telescope_types
            ):
                self.log.warning(
                    "Supplied subarray has different telescopes than subarray loaded from file"
                )
            self.__dict__.update(loaded.__dict__)

    @abstractmethod
    def __call__(self, event: ArrayEventContainer) -> None:
        """Event-wise prediction for the EventSource-Loop.

        Fills the event.dl2.<your-feature>[name] container.

        Parameters
        ----------
        event: ArrayEventContainer
        """

    @abstractmethod
    def predict_table(self, key, table: Table) -> Table:
        """
        Predict on a table of events

        Parameters
        ----------
        table : `~astropy.table.Table`
            Table of features

        Returns
        -------
        table : `~astropy.table.Table`
            Table(s) with predictions, matches the corresponding
            container definition(s)
        """

    def write(self, path):
        Provenance().add_output_file(path, role="ml-models")
        with open(path, "wb") as f:
            joblib.dump(self, f, compress=True)

    @classmethod
    def read(cls, path, **kwargs):
        with open(path, "rb") as f:
            instance = joblib.load(f)

        for attr, value in kwargs.items():
            setattr(instance, attr, value)

        if not isinstance(instance, cls):
            raise TypeError(
                f"{path} did not contain an instance of {cls}, got {instance}"
            )

        Provenance().add_input_file(path, role="ml-model")
        return instance

    @lazyproperty
    def instrument_table(self):
        return QTable(self.subarray.to_table("joined"))

    def _new_model(self):
        return SUPPORTED_MODELS[self.model_cls](**self.model_config)

    def _table_to_X(self, table):
        feature_table = table[self.features]
        valid = check_valid_rows(feature_table, log=self.log)
        X = table_to_float(feature_table[valid])
        return X, valid

    def _table_to_y(self, table, mask=None):
        if self.unit is not None:
            return table[mask][self.target].quantity.to_value(self.unit)
        return np.array(table[self.target][mask])

    def fit(self, key, table):
        """
        Create and fit a new model for ``key`` using the data in ``table``.
        """
        self._models[key] = self._new_model()

        X, valid = self._table_to_X(table)
        y = self._table_to_y(table, mask=valid)
        self.unit = table[self.target].unit
        self._models[key].fit(X, y)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_trait_values"]["parent"] = None
        state["_trait_notifiers"] = {}
        return state


class SKLearnRegressionReconstructor(SKLearnReconstructor):
    """
    Base class for regression tasks
    """

    model_cls = Enum(
        SUPPORTED_REGRESSORS.keys(),
        default_value=None,
        allow_none=True,
        help="Which scikit-learn regression model to use.",
    ).tag(config=True)

    log_target = Bool(
        default_value=False,
        help="If True, the model is trained to predict the natural logarithm.",
    ).tag(config=True)

    def _predict(self, key, table):
        if key not in self._models:
            raise KeyError(
                f"No model available for key {key},"
                f" available models: {self._models.keys()}"
            )
        X, valid = self._table_to_X(table)
        n_outputs = getattr(self._models[key], "n_outputs_", 1)

        if n_outputs > 1:
            shape = (len(table), n_outputs)
        else:
            shape = (len(table),)

        prediction = np.full(shape, np.nan)
        if np.any(valid):
            valid_predictions = self._models[key].predict(X)

            if self.log_target:
                prediction[valid] = np.exp(valid_predictions)
            else:
                prediction[valid] = valid_predictions

        if self.unit is not None:
            prediction = u.Quantity(prediction, self.unit, copy=False)

        return prediction, valid

    def _table_to_y(self, table, mask=None):
        y = super()._table_to_y(table, mask=mask)

        if self.log_target:
            if np.any(y <= 0):
                raise ValueError("y contains negative values, cannot apply log")

            return np.log(y)
        return y


class SKLearnClassficationReconstructor(SKLearnReconstructor):
    """
    Base class for classification tasks
    """

    model_cls = Enum(
        SUPPORTED_CLASSIFIERS.keys(),
        default_value=None,
        allow_none=True,
        help="Which scikit-learn regression model to use.",
    ).tag(config=True)

    invalid_class = Integer(
        default_value=-1, help="The label to fill in case no prediction could be made"
    ).tag(config=True)

    positive_class = Integer(
        default_value=1,
        help=(
            "The label value of the positive class."
            " ``prediction`` values close to 1.0 mean the event"
            " belonged likely to this class."
        ),
    ).tag(config=True)

    def _predict(self, key, table):
        if key not in self._models:
            raise KeyError(
                f"No model available for key {key},"
                f" available models: {self._models.keys()}"
            )

        X, valid = self._table_to_X(table)
        n_outputs = getattr(self._models[key], "n_outputs_", 1)

        if n_outputs > 1:
            shape = (len(table), n_outputs)
        else:
            shape = (len(table),)

        prediction = np.full(shape, self.invalid_class, dtype=np.int8)
        if np.any(valid):
            prediction[valid] = self._models[key].predict(X)

        return prediction, valid

    def _predict_score(self, key, table):
        if key not in self._models:
            raise KeyError(
                f"No model available for key {key},"
                f" available models: {self._models.keys()}"
            )

        X, valid = self._table_to_X(table)

        n_classes = getattr(self._models[key], "n_classes_", 2)
        n_rows = len(table)
        shape = (n_rows, n_classes) if n_classes > 2 else (n_rows,)

        scores = np.full(shape, np.nan)

        if np.any(valid):
            prediction = self._models[key].predict_proba(X)[:]

            if n_classes > 2:
                scores[valid] = prediction
            else:
                # only return one score for the positive class
                scores[valid] = prediction[:, self._get_positive_index(key)]

        return scores, valid

    def _get_positive_index(self, key):
        return np.nonzero(self._models[key].classes_ == self.positive_class)[0][0]


class EnergyRegressor(SKLearnRegressionReconstructor):
    """
    Use a scikit-learn regression model per telescope type to predict primary energy
    """

    #: Name of the target table column for training
    target = "true_energy"
    property = "energy"

    def __call__(self, event: ArrayEventContainer) -> None:
        """
        Apply model for a single event and fill result into the event container
        """
        for tel_id in event.trigger.tels_with_trigger:
            table = collect_features(event, tel_id, self.instrument_table)
            table = self.generate_features(table)

            passes_quality_checks = self.qualityquery.get_table_mask(table)[0]

            if passes_quality_checks:
                prediction, valid = self._predict(
                    self.subarray.tel[tel_id],
                    table,
                )
                container = ReconstructedEnergyContainer(
                    energy=prediction[0],
                    is_valid=valid[0],
                )
            else:
                container = ReconstructedEnergyContainer(
                    energy=u.Quantity(np.nan, self.unit),
                    is_valid=False,
                )

            container.prefix = f"{self.model_cls}_tel"
            event.dl2.tel[tel_id].energy[self.model_cls] = container

        self.stereo_combiner(event)

    def predict_table(self, key, table: Table) -> Table:
        """Predict on a table of events"""
        table = self.generate_features(table)

        n_rows = len(table)
        energy = u.Quantity(np.full(n_rows, np.nan), self.unit, copy=False)
        is_valid = np.full(n_rows, False)

        valid = self.qualityquery.get_table_mask(table)
        energy[valid], is_valid[valid] = self._predict(key, table[valid])

        result = Table(
            {
                f"{self.model_cls}_tel_energy": energy,
                f"{self.model_cls}_tel_is_valid": is_valid,
            }
        )
        add_defaults_and_meta(
            result,
            ReconstructedEnergyContainer,
            prefix=self.model_cls,
            stereo=False,
        )
        return result


class ParticleIdClassifier(SKLearnClassficationReconstructor):
    """
    Predict dl2 particle classification
    """

    #: Name of the target table column for training
    target = "true_shower_primary_id"
    property = "classification"

    positive_class = Integer(
        default_value=0,
        help="Particle id (in simtel system) of the positive class. Default is 0 for gammas.",
    ).tag(config=True)

    def __call__(self, event: ArrayEventContainer) -> None:
        for tel_id in event.trigger.tels_with_trigger:
            table = collect_features(event, tel_id, self.instrument_table)
            table = self.generate_features(table)
            mask = self.qualityquery.get_table_mask(table)

            if mask[0]:
                prediction, valid = self._predict_score(
                    self.subarray.tel[tel_id],
                    table,
                )

                container = ParticleClassificationContainer(
                    prediction=prediction[0],
                    is_valid=valid[0],
                )
            else:
                container = ParticleClassificationContainer(
                    prediction=np.nan, is_valid=False
                )

            container.prefix = f"{self.model_cls}_tel"
            event.dl2.tel[tel_id].classification[self.model_cls] = container

        self.stereo_combiner(event)

    def predict_table(self, key, table: Table) -> Table:
        """Predict on a table of events"""
        table = self.generate_features(table)

        n_rows = len(table)
        score = np.full(n_rows, np.nan)
        is_valid = np.full(n_rows, False)

        mask = self.qualityquery.get_table_mask(table)
        score[mask], is_valid[mask] = self._predict_score(key, table[mask])

        result = Table(
            {
                f"{self.model_cls}_tel_prediction": score,
                f"{self.model_cls}_tel_is_valid": is_valid,
            }
        )
        add_defaults_and_meta(
            result, ParticleClassificationContainer, prefix=self.model_cls, stereo=False
        )
        return result


class CrossValidator(Component):
    n_cross_validations = Int(5).tag(config=True)
    output_path = Path(
        default_value=None,
        allow_none=True,
        directory_ok=False,
    ).tag(config=True)
    rng_seed = Int(default_value=1337, help="Seed for the random number generator").tag(
        config=True
    )

    def __init__(self, model_component, **kwargs):
        super().__init__(**kwargs)
        self.cv_predictions = {}
        self.model_component = model_component
        self.rng = np.random.default_rng(self.rng_seed)

        if isinstance(self.model_component, SKLearnClassficationReconstructor):
            self.cross_validate = self._cross_validate_classification
            self.split_data = StratifiedKFold
        elif isinstance(self.model_component, SKLearnRegressionReconstructor):
            self.cross_validate = self._cross_validate_regressor
            self.split_data = KFold
        else:
            raise KeyError(
                "Unsupported Model of type %s supplied", self.model_component
            )

    def __call__(self, telescope_type, table):
        if len(table) <= self.n_cross_validations:
            raise ValueError(f"Too few events for {telescope_type}.")

        self.log.info(
            "Starting cross-validation with %d folds for type %s.",
            self.n_cross_validations,
            telescope_type,
        )

        scores = defaultdict(list)
        predictions = []

        kfold = self.split_data(
            n_splits=self.n_cross_validations,
            shuffle=True,
            # sklearn does not support numpy's new random API yet
            random_state=self.rng.integers(0, 2**31 - 1),
        )

        cv_it = kfold.split(table, table[self.model_component.target])
        for fold, (train_indices, test_indices) in enumerate(
            tqdm(
                cv_it,
                total=self.n_cross_validations,
                desc=f"Cross Validation for {telescope_type}",
            ),
        ):
            train = table[train_indices]
            test = table[test_indices]
            cv_prediction, truth, metrics = self.cross_validate(
                telescope_type, train, test
            )
            predictions.append(
                Table(
                    {
                        "cv_fold": np.full(len(truth), fold, dtype=np.uint8),
                        "tel_type": [str(telescope_type)] * len(truth),
                        "predictions": cv_prediction,
                        "truth": truth,
                    }
                )
            )

            for metric, value in metrics.items():
                scores[metric].append(value)

        for metric, cv_values in scores.items():
            cv_values = np.array(cv_values)
            with np.printoptions(precision=4):
                self.log.info(
                    "Mean %s score from CV: %.4f ± %.4f",
                    metric,
                    cv_values.mean(),
                    cv_values.std(),
                )
        self.cv_predictions[telescope_type] = vstack(predictions)

    def _cross_validate_regressor(self, telescope_type, train, test):
        regressor = self.model_component
        regressor.fit(telescope_type, train)
        prediction, _ = regressor._predict(telescope_type, test)
        truth = test[regressor.target]
        r2 = r2_score(truth, prediction)
        return prediction, truth, {"R^2": r2}

    def _cross_validate_classification(self, telescope_type, train, test):
        classifier = self.model_component
        classifier.fit(telescope_type, train)
        prediction, _ = classifier._predict_score(telescope_type, test)
        truth = np.where(
            test[classifier.target] == classifier.positive_class,
            1,
            0,
        )
        roc_auc = roc_auc_score(truth, prediction)
        return prediction, truth, {"ROC AUC": roc_auc}

    def write(self):
        Provenance().add_output_file(self.output_path, role="ml-cross-validation")
        for tel_type, results in self.cv_predictions.items():
            write_table(results, self.output_path, f"cv_predictions_{tel_type}")
