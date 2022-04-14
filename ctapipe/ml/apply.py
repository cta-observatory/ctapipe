from abc import abstractmethod

import numpy as np
from astropy.table import Table
from ctapipe.containers import ArrayEventContainer
from ctapipe.core import Component
from ctapipe.core.traits import Path

from .sklearn import Regressor


class RegressionReconstructor(Component):
    """Base class for sklearn regressors."""

    model_path = Path(
        default_value=None,
        allow_none=False,
        directory_ok=False,
    ).tag(config=True)

    # TODO: update model config (?)
    #       only settings that make sense, e.g. verbose, n_jobs

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = Regressor.load(self.model_path)

    @abstractmethod
    def __call__(self, event: ArrayEventContainer) -> None:
        """Event-wise prediction for the EventSource-Loop.

        Fill the event.dl2.<your-feature> container.

        Parameters
        ----------
        event: ArrayEventContainer
        """

    def predict(self, table: Table) -> np.array:
        """Tool"""
        return self.model.predict(table)


class EnergyRegressor(RegressionReconstructor):
    """"""

    # TODO: models per tel type
    #       but this needs to be done in the Trainer first

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, event: ArrayEventContainer) -> None:
        for tel_id in event.trigger.tels_with_trigger:
            features = self._collect_features(event, tel_id)
            prediction = self.model.predict(features)

            event.dl2.tel[tel_id].energy.energy = prediction
            event.dl2.tel[tel_id].energy.is_valid = True
            event.dl2.tel[tel_id].energy.tel_ids = [tel_id]

    def _collect_features(self, event: ArrayEventContainer, tel_id: int) -> Table:
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
