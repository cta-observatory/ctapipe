from abc import abstractmethod

import numpy as np
from astropy.table import Table
from ctapipe.containers import (
    ArrayEventContainer,
    ParticleClassificationContainer,
    ReconstructedEnergyContainer,
)
from ctapipe.core import Component

from traitlets import Instance
from .sklearn import Classifier, Regressor, Model

from typing import Tuple


class Reconstructor(Component):
    """Base class for sklearn reconstructors."""

    # TODO: update model config (?)
    #       only settings that make sense, e.g. verbose, n_jobs
    model = Instance(Model).tag(config=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def __call__(self, event: ArrayEventContainer) -> None:
        """Event-wise prediction for the EventSource-Loop.

        Fill the event.dl2.<your-feature>[name] container.

        Parameters
        ----------
        event: ArrayEventContainer
        """

    def predict(self, table: Table) -> Tuple[np.ndarray, np.ndarray]:
        """Predict on a table of events"""
        return self.model.predict(table)

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

        for container in (
            *event.dl1.tel[tel_id]
            .parameters.as_dict(add_prefix=True, recursive=True)
            .values(),
            *event.dl2.tel[tel_id].as_dict(add_prefix=True, recursive=True).values(),
        ):
            for key, value in container.items():
                features.update({key: [value]})

        return Table(features)


class RegressionReconstructor(Reconstructor):
    """Base class for sklearn regressors."""

    model = Instance(Regressor).tag(config=True)


class ClassificationReconstructor(Reconstructor):
    """Base class for sklearn regressors."""

    model = Instance(Classifier).tag(config=True)


class EnergyRegressor(RegressionReconstructor):
    """"""

    def __call__(self, event: ArrayEventContainer) -> None:
        for tel_id in event.trigger.tels_with_trigger:
            features = self._collect_features(event, tel_id)
            prediction, valid = self.model.predict(features)
            container = ReconstructedEnergyContainer(
                energy=prediction,
                is_valid=valid,
            )
            event.dl2.tel[tel_id].energy[self.model.model_cls] = container


class ParticleIdClassifier(ClassificationReconstructor):
    """"""

    def __call__(self, event: ArrayEventContainer) -> None:
        for tel_id in event.trigger.tels_with_trigger:
            features = self._collect_features(event, tel_id)
            prediction, valid = self.model.predict(features)

            container = ParticleClassificationContainer(
                prediction=prediction,
                is_valid=valid,
            )
            event.dl2.tel[tel_id].classification[self.model.model_cls] = container
