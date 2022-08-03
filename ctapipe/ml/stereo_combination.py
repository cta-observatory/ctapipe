from abc import abstractmethod

import astropy.units as u
import numpy as np
from astropy.table import Table

from ctapipe.core import Component, Container
from ctapipe.core.traits import CaselessStrEnum, Unicode

from ..containers import (
    ArrayEventContainer,
    ParticleClassificationContainer,
    ReconstructedEnergyContainer,
)


def _calculate_ufunc_of_telescope_values(tel_data, n_array_events, indices, ufunc):
    combined_values = np.zeros(n_array_events)
    ufunc.at(combined_values, indices, tel_data)
    return combined_values


def _weighted_mean_ufunc(tel_values, weights, n_array_events, indices):
    # avoid numerical problems by very large or small weights
    weights = weights / weights.max()
    sum_prediction = _calculate_ufunc_of_telescope_values(
        tel_values * weights,
        n_array_events,
        indices,
        np.add,
    )
    sum_of_weights = _calculate_ufunc_of_telescope_values(
        weights, n_array_events, indices, np.add
    )
    mean = np.full(n_array_events, np.nan)
    valid = sum_of_weights > 0
    mean[valid] = sum_prediction[valid] / sum_of_weights[valid]
    return mean


class StereoCombiner(Component):
    # TODO: Add quality query (after #1888)
    algorithm = Unicode().tag(config=True)
    combine_property = CaselessStrEnum(["energy", "classification", "direction"]).tag(
        config=True
    )

    @abstractmethod
    def __call__(self, event: ArrayEventContainer) -> None:
        """
        Fill event container with stereo predictions
        """

    @abstractmethod
    def predict(self, mono_predictions: Table) -> Table:
        """
        Constructs stereo predictions from a table of
        telescope events.
        """


class StereoMeanCombiner(StereoCombiner):
    """
    Calculates array-wide (stereo) predictions as the mean of
    the reconstruction on telescope-level with an optional weighting.
    """

    weights = CaselessStrEnum(
        ["none", "intensity", "konrad"],
        default_value="none",
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
            mono = dl2.energy[self.algorithm]
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
            mean = np.average(values, weights=weights)
            std = np.sqrt(np.cov(values, aweights=weights))
            valid = True
        else:
            mean = std = np.nan
            valid = False

        event.dl2.stereo.energy[self.algorithm] = ReconstructedEnergyContainer(
            energy=u.Quantity(mean, u.TeV, copy=False),
            energy_uncert=u.Quantity(std, u.TeV, copy=False),
            telescopes=ids,
            is_valid=valid,
        )
        event.dl2.stereo.energy[self.algorithm].prefix = self.algorithm

    def _combine_classification(self, event):
        ids = []
        values = []
        weights = []

        for tel_id, dl2 in event.dl2.tel.items():
            mono = dl2.classification[self.algorithm]
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
            prediction=mean, telescopes=ids, is_valid=valid
        )
        container.prefix = self.algorithm
        event.dl2.stereo.classification[self.algorithm] = container

    def __call__(self, event: ArrayEventContainer) -> None:
        """
        Calculate the mean prediction for a single array event.
        """
        if self.combine_property == "energy":
            self._combine_energy(event)
        elif self.combine_property == "classification":
            self._combine_classification(event)
        else:
            raise NotImplementedError(f"Cannot combine {self.combine_property}")

    def predict(self, mono_predictions: Table) -> Table:
        """
        Calculates the (array-)event-wise mean.
        Telescope events, that are nan, get discarded.
        This means you might end up with less events if
        all telescope predictions of a shower are invalid.
        """

        prefix = self.algorithm
        # TODO: Integrate table quality query once its done
        valid = mono_predictions[f"{prefix}_is_valid"]
        valid_predictions = mono_predictions[valid]

        array_events, indices, multiplicity = np.unique(
            mono_predictions[["obs_id", "event_id"]],
            return_inverse=True,
            return_counts=True,
        )
        stereo_table = Table(array_events)
        n_array_events = len(array_events)
        weights = self._calculate_weights(valid_predictions)

        if self.combine_property == "classification":
            if len(valid_predictions) > 0:
                mono_predictions = valid_predictions[f"{prefix}_prediction"]
                stereo_predictions = _weighted_mean_ufunc(
                    mono_predictions, weights, n_array_events, indices[valid]
                )
            else:
                stereo_predictions = np.full(n_array_events, np.nan)

            stereo_table[f"{prefix}_prediction"] = stereo_predictions
            stereo_table[f"{prefix}_is_valid"] = np.isfinite(stereo_predictions)
            stereo_table[f"{prefix}_goodness_of_fit"] = np.nan

        elif self.combine_property == "energy":
            if len(valid_predictions) > 0:
                mono_energies = valid_predictions[f"{prefix}_energy"].quantity.to_value(
                    u.TeV
                )
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
            else:
                stereo_energy = np.full(n_array_events, np.nan)
                variance = np.full(n_array_events, np.nan)

            stereo_table[f"{prefix}_energy"] = u.Quantity(
                stereo_energy, u.TeV, copy=False
            )

            stereo_table[f"{prefix}_energy_uncert"] = u.Quantity(
                np.sqrt(variance), u.TeV, copy=False
            )
            stereo_table[f"{prefix}_is_valid"] = np.isfinite(stereo_energy)
            stereo_table[f"{prefix}_goodness_of_fit"] = np.nan

        else:
            raise NotImplementedError()

        tel_ids = [[] for _ in range(n_array_events)]

        for index, tel_id in zip(indices[valid], valid_predictions["tel_id"]):
            tel_ids[index].append(tel_id)

        k = f"{prefix}_tel_ids"
        stereo_table[k] = tel_ids
        return stereo_table
