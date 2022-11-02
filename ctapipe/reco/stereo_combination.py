from abc import abstractmethod

import astropy.units as u
import numpy as np
from astropy.table import Table

from ctapipe.core import Component, Container
from ctapipe.core.traits import Bool, CaselessStrEnum, Unicode

from ..containers import (
    ArrayEventContainer,
    ParticleClassificationContainer,
    ReconstructedEnergyContainer,
)
from .utils import add_defaults_and_meta

_containers = {
    "energy": ReconstructedEnergyContainer,
    "classification": ParticleClassificationContainer,
}

__all__ = [
    "StereoCombiner",
    "StereoMeanCombiner",
]


def _grouped_add(tel_data, n_array_events, indices):
    """
    Calculate the group-wise sum for each array event over the
    corresponding telescope events. ``indices`` is an array
    that gives the index of the subarray event for each telescope event,
    returned by
    ``np.unique(tel_events[["obs_id", "event_id"]], return_inverse=True)``
    """
    combined_values = np.zeros(n_array_events)
    np.add.at(combined_values, indices, tel_data)
    return combined_values


def _weighted_mean_ufunc(tel_values, weights, n_array_events, indices):
    # avoid numerical problems by very large or small weights
    weights = weights / weights.max()
    sum_prediction = _grouped_add(
        tel_values * weights,
        n_array_events,
        indices,
    )
    sum_of_weights = _grouped_add(
        weights,
        n_array_events,
        indices,
    )
    mean = np.full(n_array_events, np.nan)
    valid = sum_of_weights > 0
    mean[valid] = sum_prediction[valid] / sum_of_weights[valid]
    return mean


class StereoCombiner(Component):
    """Base Class for algorithms combining telescope-wise predictions to common prediction"""

    prefix = Unicode(
        default_value="",
        help="Prefix to be added to the output container / column names",
    ).tag(config=True)

    property = CaselessStrEnum(
        ["energy", "classification", "geometry"],
        help="Which property is being combined",
    ).tag(config=True)

    @abstractmethod
    def __call__(self, event: ArrayEventContainer) -> None:
        """
        Fill event container with stereo predictions
        """

    @abstractmethod
    def predict_table(self, mono_predictions: Table) -> Table:
        """
        Constructs stereo predictions from a table of
        telescope events.
        """


class StereoMeanCombiner(StereoCombiner):
    """
    Calculate array-event prediction as (weighted) mean of telescope-wise predictions
    """

    weights = CaselessStrEnum(
        ["none", "intensity", "konrad"],
        default_value="none",
    ).tag(config=True)

    log_target = Bool(
        False,
        help="If true, calculate exp(mean(log(values)))",
    ).tag(config=True)

    def _calculate_weights(self, data):
        """"""

        if isinstance(data, Container):
            if self.weights == "intensity":
                return data.hillas.intensity

            if self.weights == "konrad":
                return data.hillas.intensity * data.hillas.length / data.hillas.width

            return 1

        if isinstance(data, Table):
            if self.weights == "intensity":
                return data["hillas_intensity"]

            if self.weights == "konrad":
                return (
                    data["hillas_intensity"]
                    * data["hillas_length"]
                    / data["hillas_width"]
                )

            return np.ones(len(data))

        raise TypeError(
            "Dl1 data needs to be provided in the form of a container or astropy.table.Table"
        )

    def _combine_energy(self, event):
        ids = []
        values = []
        weights = []

        for tel_id, dl2 in event.dl2.tel.items():
            mono = dl2.energy[self.prefix]
            if mono.is_valid:
                values.append(mono.energy.to_value(u.TeV))
                if tel_id not in event.dl1.tel:
                    raise ValueError("No parameters for weighting available")
                weights.append(
                    self._calculate_weights(event.dl1.tel[tel_id].parameters)
                )
                ids.append(tel_id)

        if len(values) > 0:
            weights = np.array(weights, dtype=np.float64)
            weights /= weights.max()

            if self.log_target:
                values = np.log(values)

            mean = np.average(values, weights=weights)
            std = np.sqrt(np.cov(values, aweights=weights))

            if self.log_target:
                mean = np.exp(mean)
                std = np.exp(std)

            valid = True
        else:
            mean = std = np.nan
            valid = False

        event.dl2.stereo.energy[self.prefix] = ReconstructedEnergyContainer(
            energy=u.Quantity(mean, u.TeV, copy=False),
            energy_uncert=u.Quantity(std, u.TeV, copy=False),
            telescopes=ids,
            is_valid=valid,
            prefix=self.prefix,
        )

    def _combine_classification(self, event):
        ids = []
        values = []
        weights = []

        for tel_id, dl2 in event.dl2.tel.items():
            mono = dl2.classification[self.prefix]
            if mono.is_valid:
                values.append(mono.prediction)
                dl1 = event.dl1.tel[tel_id].parameters
                weights.append(self._calculate_weights(dl1) if dl1 else 1)
                ids.append(tel_id)

        if len(values) > 0:
            mean = np.average(values, weights=weights)
            valid = True
        else:
            mean = np.nan
            valid = False

        container = ParticleClassificationContainer(
            prediction=mean, telescopes=ids, is_valid=valid, prefix=self.prefix
        )
        event.dl2.stereo.classification[self.prefix] = container

    def __call__(self, event: ArrayEventContainer) -> None:
        """
        Calculate the mean prediction for a single array event.
        """
        if self.property == "energy":
            self._combine_energy(event)
        elif self.property == "classification":
            self._combine_classification(event)
        else:
            raise NotImplementedError(f"Cannot combine {self.property}")

    def predict_table(self, mono_predictions: Table) -> Table:
        """
        Calculates the (array-)event-wise mean.
        Telescope events, that are nan, get discarded.
        This means you might end up with less events if
        all telescope predictions of a shower are invalid.
        """

        prefix = f"{self.prefix}_tel"
        # TODO: Integrate table quality query once its done
        valid = mono_predictions[f"{prefix}_is_valid"]
        valid_predictions = mono_predictions[valid]

        array_events, indices, multiplicity = np.unique(
            mono_predictions[["obs_id", "event_id"]],
            return_inverse=True,
            return_counts=True,
        )
        stereo_table = Table(array_events)
        # copy metadata
        for colname in ("obs_id", "event_id"):
            stereo_table[colname].description = mono_predictions[colname].description

        n_array_events = len(array_events)
        weights = self._calculate_weights(valid_predictions)

        if self.property == "classification":
            if len(valid_predictions) > 0:
                mono_predictions = valid_predictions[f"{prefix}_prediction"]
                stereo_predictions = _weighted_mean_ufunc(
                    mono_predictions, weights, n_array_events, indices[valid]
                )
            else:
                stereo_predictions = np.full(n_array_events, np.nan)

            stereo_table[f"{self.prefix}_prediction"] = stereo_predictions
            stereo_table[f"{self.prefix}_is_valid"] = np.isfinite(stereo_predictions)
            stereo_table[f"{self.prefix}_goodness_of_fit"] = np.nan

        elif self.property == "energy":
            if len(valid_predictions) > 0:
                mono_energies = valid_predictions[f"{prefix}_energy"].quantity.to_value(
                    u.TeV
                )

                if self.log_target:
                    mono_energies = np.log(mono_energies)

                stereo_energy = _weighted_mean_ufunc(
                    mono_energies,
                    weights,
                    n_array_events,
                    indices[valid],
                )
                variance = _weighted_mean_ufunc(
                    (mono_energies - np.repeat(stereo_energy, multiplicity)[valid])
                    ** 2,
                    weights,
                    n_array_events,
                    indices[valid],
                )
                std = np.sqrt(variance)

                if self.log_target:
                    stereo_energy = np.exp(stereo_energy)
                    std = np.exp(std)
            else:
                stereo_energy = np.full(n_array_events, np.nan)
                std = np.full(n_array_events, np.nan)

            stereo_table[f"{self.prefix}_energy"] = u.Quantity(
                stereo_energy, u.TeV, copy=False
            )

            stereo_table[f"{self.prefix}_energy_uncert"] = u.Quantity(
                std, u.TeV, copy=False
            )
            stereo_table[f"{self.prefix}_is_valid"] = np.isfinite(stereo_energy)
            stereo_table[f"{self.prefix}_goodness_of_fit"] = np.nan
        else:
            raise NotImplementedError()

        tel_ids = [[] for _ in range(n_array_events)]

        for index, tel_id in zip(indices[valid], valid_predictions["tel_id"]):
            tel_ids[index].append(tel_id)

        stereo_table[f"{self.prefix}_telescopes"] = tel_ids
        add_defaults_and_meta(stereo_table, _containers[self.property], self.prefix)
        return stereo_table
