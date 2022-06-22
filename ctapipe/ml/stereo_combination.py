from abc import abstractmethod
import numpy as np
from astropy.table import Table
import astropy.units as u
from scipy.ndimage import median
from ctapipe.core import Component
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


class StereoCombiner(Component):
    # TODO: Add quality query (after #1888)
    def __init__(self, algorithm, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.algorithm = algorithm

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

    def check_valid(self, table: Table) -> np.ndarray:
        """
        Selects the telescope events, that should be used
        for the stereo reconstruction.
        Due to the quality query only operating on single events currently,
        as of now this only discards rows with nans in them.
        """
        valid = check_valid_rows(table)
        # TODO: Add quality query (after #1888)
        return valid


class StereoMeanCombiner(StereoCombiner):
    """
    Calculates array-wide (stereo) predictions as the mean of
    the reconstruction on telescope-level with an optional weighting.
    """

    def __init__(self, weight_column=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_column = weight_column

    def _construct_weights(self, table):
        # TODO: Do we want to calculate weights on the fly or do it externally?
        # If we dont want to do that here, this might as well be removed
        if self.weight_column:
            return table[self.weight_column]
        else:
            return None

    def __call__(self, event: ArrayEventContainer) -> None:
        """
        Calculate the mean energy / classification prediction for a single
        array event.
        TODO: This only uses the nominal telescope values, not their respective uncertainties!
        (Which we dont have right now)
        """
        mono_energies = {"ids": [], "values": [], "weights": None}
        mono_classifications = {"ids": [], "values": [], "weights": None}

        for tel_id, dl2 in event.dl2.tel.items():
            if self.algorithm in dl2.energy:
                mono = dl2.energy[self.algorithm]
                if mono.is_valid:
                    mono_energies["values"].append(mono.energy.to_value(u.TeV))
                    mono_energies["ids"].append(tel_id)
            if self.algorithm in dl2.classification:
                mono = dl2.classification[self.algorithm]
                if mono.is_valid:
                    mono_classifications["values"].append(mono.prediction)
                    mono_classifications["ids"].append(tel_id)
        if mono_energies["ids"]:
            weighted_mean = np.average(
                mono_energies["values"], weights=mono_energies["weights"]
            )
            stereo_energy = ReconstructedEnergyContainer(
                energy=u.Quantity(weighted_mean, u.TeV, copy=False),
                energy_uncert=u.Quantity(
                    np.average((weighted_mean - mono_energies["values"]) ** 2),
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

        valid_rows = self.check_valid(mono_predictions)
        valid_predictions = mono_predictions[valid_rows]
        prediction_unit = valid_predictions[self.mono_prediction_column].unit

        array_events, indices, n_tels_per_event = np.unique(
            valid_predictions[["obs_id", "event_id"]],
            return_inverse=True,
            return_counts=True,
        )
        stereo_table = Table(array_events)
        n_array_events = len(array_events)
        weights = self._construct_weights(valid_predictions)

        if weights is not None:
            sum_prediction = _calculate_ufunc_of_telescope_values(
                valid_predictions[self.mono_prediction_column] * weights,
                n_array_events,
                indices,
                np.add,
            )
            sum_of_weights = _calculate_ufunc_of_telescope_values(
                weights, n_array_events, indices, np.add
            )
        else:
            sum_prediction = _calculate_ufunc_of_telescope_values(
                valid_predictions[self.mono_prediction_column],
                n_array_events,
                indices,
                np.add,
            )
            sum_of_weights = n_tels_per_event
        stereo_table[f"mean_{self.mono_prediction_column}"] = u.Quantity(
            sum_prediction / sum_of_weights, prediction_unit, copy=False
        )
        return stereo_table
