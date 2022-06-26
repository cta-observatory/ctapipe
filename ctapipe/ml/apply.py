from abc import abstractmethod
from typing import Tuple

import numpy as np
from astropy.table import Table
from traitlets import Instance

from ..containers import (
    ArrayEventContainer,
    ParticleClassificationContainer,
    ReconstructedEnergyContainer,
)
from ..core import Component
from .sklearn import Classifier, Regressor, Model


__all__ = [
    "Reconstructor",
    "ClassificationReconstructor",
    "RegressionReconstructor",
    "EnergyRegressor",
    "ParticleIdClassifier",
]


class Reconstructor(Component):
    """Base class for sklearn reconstructors."""

    # TODO: update model config (?)
    #       only settings that make sense, e.g. verbose, n_jobs
    model_cls = Model
    model = Instance(Model).tag(config=True)

    def __init__(self, subarray, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subarray = subarray
        self.instrument_table = self.subarray.to_table("joined")

    @classmethod
    def read(cls, path, subarray, *args, **kwargs):
        model = cls.model_cls.load(path)
        return cls(subarray, *args, model=model, **kwargs)

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
        ):
            features.update(container)

        # for key, container in event.dl2.tel[tel_id]:

        for containers in event.dl2.stereo.values():
            for algorithm, container in containers.items():
                prefix = container.prefix
                if prefix:
                    container.prefix = f"{algorithm}_{prefix}"
                else:
                    container.prefix = algorithm

                features.update(container.as_dict(add_prefix=True))
                container.prefix = ""

        features.update(self.instrument_table.loc[tel_id])
        return Table({k: [v] for k, v in features.items()})


class RegressionReconstructor(Reconstructor):
    """Base class for sklearn regressors."""

    model_cls = Regressor
    model = Instance(model_cls).tag(config=True)


class ClassificationReconstructor(Reconstructor):
    """Base class for sklearn regressors."""

    model_cls = Classifier
    model = Instance(model_cls).tag(config=True)


class EnergyRegressor(RegressionReconstructor):
    """"""

    def __call__(self, event: ArrayEventContainer) -> None:
        for tel_id in event.trigger.tels_with_trigger:
            features = self._collect_features(event, tel_id)
            prediction, valid = self.model.predict(features)
            container = ReconstructedEnergyContainer(
                energy=prediction[0],
                is_valid=valid[0],
            )
            event.dl2.tel[tel_id].energy[self.model.model_cls] = container


class ParticleIdClassifier(ClassificationReconstructor):
    """"""

    def __call__(self, event: ArrayEventContainer) -> None:
        for tel_id in event.trigger.tels_with_trigger:
            features = self._collect_features(event, tel_id)
            prediction, valid = self.model.predict(features)

            container = ParticleClassificationContainer(
                prediction=prediction[0],
                is_valid=valid[0],
            )
            event.dl2.tel[tel_id].classification[self.model.model_cls] = container
