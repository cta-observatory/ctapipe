from abc import abstractmethod
from collections import defaultdict

import astropy.units as u
import joblib
import numpy as np
from astropy.table import Table, vstack
from astropy.utils.decorators import lazyproperty
from sklearn import metrics
from sklearn.model_selection import KFold
from tqdm import tqdm
from traitlets import Instance

from ctapipe.core.traits import Int, Path
from ctapipe.io import write_table

from ..containers import (
    ArrayEventContainer,
    ParticleClassificationContainer,
    ReconstructedEnergyContainer,
)
from ..core import Component, FeatureGenerator, Provenance, QualityQuery
from .sklearn import Classifier, Model, Regressor

__all__ = [
    "Reconstructor",
    "ClassificationReconstructor",
    "RegressionReconstructor",
    "EnergyRegressor",
    "ParticleIdClassifier",
]


class Reconstructor(Component):
    """Base class for sklearn reconstructors."""

    target = None
    model_cls = Model
    model = Instance(model_cls, allow_none=True).tag(config=True)

    def __init__(self, subarray, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subarray = subarray
        self.qualityquery = QualityQuery(parent=self)
        self.generate_features = FeatureGenerator(parent=self)

        if self.model is None:
            self.model = self.model_cls(parent=self, target=self.target)

    def write(self, path):
        Provenance().add_output_file(path, role="ml-model")
        with open(path, "wb") as f:
            joblib.dump(
                (
                    self.model,
                    self.qualityquery.quality_criteria,
                    self.generate_features.features,
                    self.subarray,
                ),
                f,
                compress=True,
            )

    @classmethod
    def read(cls, path, check_cls=True, **kwargs):
        with open(path, "rb") as f:
            model, quality_criteria, gen_features, subarray = joblib.load(f)

        if check_cls is True and model.__class__ is not cls.model_cls:
            raise TypeError(
                f"File did not contain an instance of {cls.model_cls}, got {model.__class__}"
            )

        Provenance().add_input_file(path, role="ml-model")
        instance = cls(subarray=subarray, model=model, **kwargs)
        instance.qualityquery = QualityQuery(
            quality_criteria=quality_criteria, parent=instance
        )
        instance.generate_features = FeatureGenerator(
            features=gen_features, parent=instance
        )
        return instance

    @lazyproperty
    def instrument_table(self):
        return self.subarray.to_table("joined")

    @abstractmethod
    def __call__(self, event: ArrayEventContainer) -> None:
        """Event-wise prediction for the EventSource-Loop.

        Fills the event.dl2.<your-feature>[name] container.

        Parameters
        ----------
        event: ArrayEventContainer
        """

    @abstractmethod
    def predict(self, key, table: Table) -> Table:
        """
        Predict on a table of events

        Parameters
        ----------
        table : `~astropy.table.Table`
            Table of features

        Returns
        -------
        table : `~astropy.table.Table`
            Table with predictions, matches the corresponding
            container definition
        """

    def _collect_features(self, event: ArrayEventContainer, tel_id: int) -> Table:
        """Loop over all containers with features.

        Parameters
        ----------
        event: ArrayEventContainer

        Returns
        -------
        Table
        """
        features = dict()

        features.update(
            event.dl1.tel[tel_id].parameters.as_dict(
                add_prefix=True,
                recursive=True,
                flatten=True,
            )
        )

        features.update(
            event.dl2.tel[tel_id].as_dict(
                add_prefix=False,  # would duplicate prefix, as this is part of the name of the container
                recursive=True,
                flatten=True,
            )
        )

        features.update(
            event.dl2.stereo.as_dict(
                add_prefix=False,  # see above
                recursive=True,
                flatten=True,
            )
        )

        features.update(self.instrument_table.loc[tel_id])

        return Table({k: [v] for k, v in features.items()})


class RegressionReconstructor(Reconstructor):
    """Base class for sklearn regressors."""

    model_cls = Regressor
    model = Instance(model_cls, allow_none=True).tag(config=True)


class ClassificationReconstructor(Reconstructor):
    """Base class for sklearn regressors."""

    model_cls = Classifier
    model = Instance(model_cls, allow_none=True).tag(config=True)


class EnergyRegressor(RegressionReconstructor):
    """
    Predict dl2 energy for each telescope
    """

    target = "true_energy"

    def __call__(self, event: ArrayEventContainer) -> None:
        """
        Apply the quality query and model and fill the corresponding container
        """
        for tel_id in event.trigger.tels_with_trigger:
            table = self._collect_features(event, tel_id)
            table = self.generate_features(table)
            mask = self.qualityquery.get_table_mask(table)

            if mask[0]:
                prediction, valid = self.model.predict(
                    self.subarray.tel[tel_id],
                    table,
                )
                container = ReconstructedEnergyContainer(
                    energy=prediction[0],
                    is_valid=valid[0],
                )
            else:
                container = ReconstructedEnergyContainer(
                    energy=u.Quantity(np.nan, self.model.unit),
                    is_valid=False,
                )
            event.dl2.tel[tel_id].energy[self.model.model_cls] = container

    def predict(self, key, table: Table) -> Table:
        """Predict on a table of events"""
        table = self.generate_features(table)

        n_rows = len(table)
        energy = u.Quantity(np.full(n_rows, np.nan), self.model.unit, copy=False)
        is_valid = np.full(n_rows, False)

        mask = self.qualityquery.get_table_mask(table)
        energy[mask], is_valid[mask] = self.model.predict(key, table[mask])

        result = Table(
            {
                f"{self.model.model_cls}_energy": energy,
                f"{self.model.model_cls}_is_valid": is_valid,
            }
        )
        return result


class ParticleIdClassifier(ClassificationReconstructor):
    """
    Predict dl2 particle classification
    """

    target = "true_shower_primary_id"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # gammas have true_shower_primary_id = 0
        self.model.positive_class = 0

    def __call__(self, event: ArrayEventContainer) -> None:
        for tel_id in event.trigger.tels_with_trigger:
            table = self._collect_features(event, tel_id)
            table = self.generate_features(table)
            mask = self.qualityquery.get_table_mask(table)

            if mask[0]:
                prediction, valid = self.model.predict_score(
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

            event.dl2.tel[tel_id].classification[self.model.model_cls] = container

    def predict(self, key, table: Table) -> Table:
        """Predict on a table of events"""
        table = self.generate_features(table)

        n_rows = len(table)
        score = np.full(n_rows, np.nan)
        is_valid = np.full(n_rows, False)

        mask = self.qualityquery.get_table_mask(table)
        score[mask], is_valid[mask] = self.model.predict_score(key, table[mask])

        result = Table(
            {
                f"{self.model.model_cls}_prediction": score,
                f"{self.model.model_cls}_is_valid": is_valid,
            }
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
        if isinstance(self.model_component, ClassificationReconstructor):
            self.calculate_metrics = self._cross_validate_classification
        elif isinstance(self.model_component, RegressionReconstructor):
            self.calculate_metrics = self._cross_validate_energy
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

        kfold = KFold(
            n_splits=self.n_cross_validations,
            shuffle=True,
            # sklearn does not support numpy's new random API yet
            random_state=self.rng.integers(0, 2**31 - 1),
        )

        for fold, (train_indices, test_indices) in enumerate(
            tqdm(kfold.split(table), total=self.n_cross_validations)
        ):
            train = table[train_indices]
            test = table[test_indices]
            cv_prediction, truth, metrics = self.calculate_metrics(
                telescope_type, train, test
            )
            predictions.append(
                Table(
                    {
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
                    "Mean % score from CV: %.4f ± %.4f",
                    metric,
                    cv_values.mean(),
                    cv_values.std(),
                )
        self.cv_predictions[telescope_type] = vstack(predictions)

    def _cross_validate_energy(self, telescope_type, train, test):
        regressor = self.model_component
        regressor.model.fit(telescope_type, train)
        prediction, _ = regressor.model.predict(telescope_type, test)
        truth = test[regressor.target]
        r2 = metrics.r2_score(truth, prediction)
        return prediction, truth, {"R^2": r2}

    def _cross_validate_classification(self, telescope_type, train, test):
        classifier = self.model_component
        classifier.model.fit(telescope_type, train)
        prediction, _ = classifier.model.predict_score(telescope_type, test)
        truth = np.where(
            test[classifier.target] == classifier.model.positive_class,
            1,
            0,
        )
        roc_auc = metrics.roc_auc_score(truth, prediction)
        return prediction, truth, {"ROC AUC": roc_auc}

    def write(self):
        Provenance().add_output_file(self.output_path, role="ml-cross-validation")
        for tel_type, results in self.cv_predictions.items():
            print(results, type(results))
            write_table(results, self.output_path, f"cv_predictions_{tel_type}")
