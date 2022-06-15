from abc import abstractmethod
import numpy as np
from astropy.table import Table
from ctapipe.core import Component
from ctapipe.ml.preprocessing import check_valid_rows


def _create_array_tel_mapping(tel_events_table):
    unique, indices, n_tels_per_event = np.unique(
        tel_events_table[["obs_id", "event_id"]],
        return_inverse=True,
        return_counts=True,
    )
    return len(unique), n_tels_per_event, indices


def _calculate_ufunc_of_telescope_values(tel_data, n_array_events, indices, ufunc):
    mean_values = np.zeros(n_array_events)
    ufunc.at(mean_values, indices, tel_data)
    return mean_values


class StereoCombiner(Component):
    # TODO: Add quality query (after #1888)
    def __init__(self, mono_prediction_column, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mono_prediction_column = mono_prediction_column

    @abstractmethod
    def __call__(self, mono_predictions: Table) -> np.ndarray:
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

    def __init__(self, mono_prediction_column, weight_column=None, *args, **kwargs):
        super().__init__(mono_prediction_column=mono_prediction_column, *args, **kwargs)
        self.weight_column = weight_column

    def _construct_weights(self, table):
        # TODO: Do we want to calculate weights on the fly or do it externally?
        # If we dont want to do that here, this might as well be removed
        if self.weight_column:
            return table[self.weight_column]
        else:
            return None

    def __call__(self, mono_predictions: Table) -> Table:
        """
        Calculates the (array-)event-wise mean of
        `mono_prediction_column`.
        Telescope events, that are nan, get discarded.
        This means you might end up with less events if
        all telescope predictions of a shower are invalid.
        """
        valid_rows = self.check_valid(mono_predictions)
        valid_predictions = mono_predictions[valid_rows]
        n_array_events, n_tels_per_event, indices = _create_array_tel_mapping(
            valid_predictions
        )
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
        return sum_prediction / sum_of_weights
