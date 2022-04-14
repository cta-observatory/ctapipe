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

    def __call__(self, event: ArrayEventContainer) -> None:
        return None


class EnergyRegressor(RegressionReconstructor):
    """"""

    # TODO: models per tel type
    #       but this needs to be done in the Trainer first

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
