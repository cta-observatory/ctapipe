import numpy as np
from astropy.table import Table
from ctapipe.containers import ArrayEventContainer, ReconstructedEnergyContainer
from ctapipe.core import Component

from .sklearn import Regressor


class EnergyReconstructor(Component):
    """Base class for energy estimation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, event: ArrayEventContainer) -> ReconstructedEnergyContainer:
        return ReconstructedEnergyContainer()


class EnergyRegressor(EnergyReconstructor):
    """sklearn-based energy regression"""

    # TODO: models per tel type
    #       but this needs to be done in the Trainer first

    def __init__(self, features, model_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = Regressor.load(model_path)
        # TODO: update model config
        self.model.model.n_jobs = -1
        self.model.model.verbose = 0
        # TODO: use config system
        self.features = features

    def __call__(self, event: ArrayEventContainer) -> ReconstructedEnergyContainer:
        """EventSource Loop"""
        for tel_id in event.trigger.tels_with_trigger:
            features = dict()

            for name, container in event.dl1.tel[tel_id].parameters.as_dict().items():
                for key, value in container.items():
                    # TODO: find a better way of creating these values (prefix?)
                    features.update({f"{name}_{key}": [value]})
            # TODO: get all possible features

            feature_array = Table(features)

            # TODO: since a single nan is enough for a single row to be discarded,
            #       find it earlier for small performance reasons
            if np.count_nonzero(feature_array.to_pandas().isna()):
                continue
            else:
                prediction = self.model.predict(feature_array)
                event.dl2.tel[tel_id].energy.energy = prediction
                event.dl2.tel[tel_id].energy.is_valid = True
                event.dl2.tel[tel_id].energy.tel_ids = [tel_id]

    def predict(self, table: Table) -> Table:
        """"""
        raise NotImplementedError
