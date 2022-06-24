from abc import abstractmethod
from ctapipe.core import traits
import numpy as np
from astropy.table import Table
import astropy.units as u
from scipy.ndimage import median
from ctapipe.core import Component, Container
from ctapipe.core.traits import CaselessStrEnum, Unicode
from ctapipe.ml.preprocessing import check_valid_rows
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
    sum_prediction = _calculate_ufunc_of_telescope_values(
        tel_values * weights,
        n_array_events,
        indices,
        np.add,
    )
    sum_of_weights = _calculate_ufunc_of_telescope_values(
        weights, n_array_events, indices, np.add
    )
    return sum_prediction / sum_of_weights


class StereoCombiner(Component):
    # TODO: Add quality query (after #1888)
    algorithm = Unicode().tag(config=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def __call__(self, event: ArrayEventContainer) -> None:
        """
        Fill event container with stereo predictions
        """

    @abstractmethod
    def predict(self, mono_predictions: Table) -> np.ndarray:
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
        ["none", "intensity", "konrad"], default_value="none"
    ).tag(config=True)
    combine_property = CaselessStrEnum(["energy", "classification", "direction"]).tag(
        config=True
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _calculate_weights(self, dl1):
        """ """
        if isinstance(dl1, Container):
            if self.weights == "intensity":
                return dl1.hillas.intensity
            elif self.weights == "konrad":
                return dl1.hillas.intensity * dl1.hillas.length / dl1.hillas.width
            else:
                return 1
        elif isinstance(dl1, Table):
            if self.weights == "intensity":
                return dl1["hillas_intensity"]
            elif self.weights == "konrad":
                return (
                    dl1["hillas_intensity"] * dl1["hillas_length"] / dl1["hillas_width"]
                )
            else:
                return np.ones(len(dl1))
        else:
            print(type(dl1))
            raise TypeError(
                "Dl1 data needs to be provided in the form of a container or astropy.table.Table"
            )

    def __call__(self, event: ArrayEventContainer) -> None:
        """
        Calculate the mean energy / classification prediction for a single
        array event.
        TODO: This only uses the nominal telescope values, not their respective uncertainties!
        (Which we dont have right now)
        """
        mono_energies = {"ids": [], "values": [], "weights": []}
        mono_classifications = {"ids": [], "values": [], "weights": []}

        for tel_id, dl2 in event.dl2.tel.items():
            dl1 = event.dl1.tel[tel_id].parameters
            if self.combine_property == "energy":
                mono = dl2.energy[self.algorithm]
                if mono.is_valid:
                    mono_energies["values"].append(mono.energy.to_value(u.TeV))
                    mono_energies["weights"].append(
                        self._calculate_weights(dl1) if dl1 else 1
                    )
                    mono_energies["ids"].append(tel_id)
            if self.combine_property == "classification":
                mono = dl2.classification[self.algorithm]
                if mono.is_valid:
                    mono_classifications["values"].append(mono.prediction)
                    mono_classifications["weights"].append(
                        self._calculate_weights(dl1) if dl1 else 1
                    )
                    mono_classifications["ids"].append(tel_id)
        if mono_energies["ids"]:
            weighted_mean = np.average(
                mono_energies["values"], weights=mono_energies["weights"]
            )
            stereo_energy = ReconstructedEnergyContainer(
                energy=u.Quantity(weighted_mean, u.TeV, copy=False),
                energy_uncert=u.Quantity(
                    np.average(
                        (weighted_mean - mono_energies["values"]) ** 2,
                        weights=mono_energies["weights"],
                    ),
                    u.TeV,
                    copy=False,
                ),
                tel_ids=mono_energies["ids"],
            )
            event.dl2.stereo.energy[self.algorithm] = stereo_energy
        if mono_classifications["ids"]:
            stereo_classification = ParticleClassificationContainer(
                prediction=np.average(
                    mono_classifications["values"],
                    weights=mono_classifications["weights"],
                ),
                tel_ids=mono_classifications["ids"],
            )
            event.dl2.stereo.classification[self.algorithm] = stereo_classification

    def predict(self, mono_predictions: Table) -> Table:
        """
        Calculates the (array-)event-wise mean of
        `mono_prediction_column`.
        Telescope events, that are nan, get discarded.
        This means you might end up with less events if
        all telescope predictions of a shower are invalid.
        """

        if self.combine_property == "energy":
            column_prefix = f"{self.algorithm}_reconstructed_energy"
            # TODO: Integrate table quality query once its done #1888
            valid_rows = mono_predictions[f"{column_prefix}_is_valid"]
            valid_predictions = mono_predictions[valid_rows]

            array_events, split_index, indices = np.unique(
                valid_predictions[["obs_id", "event_id"]],
                return_inverse=True,
                return_index=True,
            )
            stereo_table = Table(array_events)
            n_array_events = len(array_events)

            weights = self._calculate_weights(valid_predictions)

            stereo_table[f"{column_prefix}_tel_ids"] = np.split(
                valid_predictions["tel_id"].value, split_index
            )[1:]
            mono_energies = valid_predictions[
                f"{column_prefix}_energy"
            ].quantity.to_value(u.TeV)
            stereo_energy = _weighted_mean_ufunc(
                mono_energies, weights, n_array_events, indices
            )
            stereo_table[f"{column_prefix}_energy"] = u.Quantity(
                stereo_energy, u.TeV, copy=False
            )

            # This subtracts the array-event-wise mean from each telescope event
            # The split works the same as for the tel_ids above, but since the groups are of uneven
            # size, numpy does not allow them to be in one array without turning the dtype to object,
            # which is why the subtraction is performed in each array individually
            # Akward arrays might fit here
            centered_mono_energies = np.concatenate(
                [
                    tel - mean
                    for tel, mean in zip(
                        np.split(mono_energies, split_index)[1:], stereo_energy
                    )
                ]
            )
            stereo_energy_uncert = np.sqrt(
                _weighted_mean_ufunc(
                    centered_mono_energies**2, weights, n_array_events, indices
                )
            )
            stereo_table[f"{column_prefix}_energy_uncert"] = u.Quantity(
                stereo_energy_uncert, u.TeV, copy=False
            )
            stereo_table[f"{column_prefix}_is_valid"] = np.isfinite(stereo_energy)
            stereo_table[f"{column_prefix}_goodness_of_fit"] = np.nan

        elif self.combine_property == "classification":
            column_prefix = f"{self.algorithm}_particle_classification"

            # TODO: Integrate table quality query once its done #1888
            valid_rows = mono_predictions[f"{column_prefix}_is_valid"]
            valid_predictions = mono_predictions[valid_rows]

            array_events, split_index, indices = np.unique(
                valid_predictions[["obs_id", "event_id"]],
                return_inverse=True,
                return_index=True,
            )
            stereo_table = Table(array_events)
            n_array_events = len(array_events)
            weights = self._calculate_weights(valid_predictions)

            stereo_table[f"{column_prefix}_tel_ids"] = np.split(
                valid_predictions["tel_id"].value, split_index
            )[1:]
            mono_predictions = valid_predictions[f"{column_prefix}_prediction"]
            stereo_predictions = _weighted_mean_ufunc(
                mono_predictions, weights, n_array_events, indices
            )
            stereo_table[f"{column_prefix}_prediction"] = stereo_predictions
            stereo_table[f"{column_prefix}_is_valid"] = (stereo_predictions >= 0) & (
                stereo_predictions <= 1
            )
            stereo_table[f"{column_prefix}_goodness_of_fit"] = np.nan

        else:
            raise NotImplementedError()

        return stereo_table
