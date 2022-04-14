from astropy.table import Table
from ctapipe.containers import ArrayEventContainer
from ctapipe.core import Component

from .sklearn import Regressor


class EnergyReconstructor(Component):
    """Base class for energy estimation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, event: ArrayEventContainer) -> None:
        return None


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




    def __call__(self, event: ArrayEventContainer) -> None:
        """EventSource Loop"""
        for tel_id in event.trigger.tels_with_trigger:
            features = dict()

            for container in (
                *event.dl1.tel[tel_id]
                .parameters.as_dict(add_prefix=True, recursive=True)
                .values(),
                *event.dl2.tel[tel_id]
                .as_dict(add_prefix=True, recursive=True)
                .values(),
            ):
                for key, value in container.items():
                    features.update({key: [value]})

            feature_array = Table(features)

            prediction = self.model.predict(feature_array)
            event.dl2.tel[tel_id].energy.energy = prediction
            event.dl2.tel[tel_id].energy.is_valid = True
            event.dl2.tel[tel_id].energy.tel_ids = [tel_id]

    def predict(self, table: Table) -> Table:
        """"""
        raise NotImplementedError
