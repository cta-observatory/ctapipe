"""Helper functions for array-event-wise aggregation of telescope events."""

from collections import namedtuple

import numpy as np
from numba import njit, uint64

from ctapipe.core.env import CTAPIPE_DISABLE_NUMBA_CACHE

__all__ = ["get_subarray_index", "weighted_mean_std_ufunc"]


SubarrayIndex = namedtuple(
    "SubarrayIndex",
    [
        "obs_id",
        "event_id",
        "multiplicity",
        "first_tel_event_index",
        "subarray_event_index",
    ],
)


@njit(cache=not CTAPIPE_DISABLE_NUMBA_CACHE)
def _get_subarray_index(obs_ids, event_ids):
    n_tel_events = len(obs_ids)
    idx = np.zeros(n_tel_events, dtype=uint64)
    current_idx = 0
    subarray_obs_ids = []
    subarray_event_ids = []
    subarray_first_index = []

    multiplicities = []
    multiplicity = 0

    if n_tel_events > 0:
        subarray_obs_ids.append(obs_ids[0])
        subarray_event_ids.append(event_ids[0])
        subarray_first_index.append(0)
        multiplicity += 1

    for i in range(1, n_tel_events):
        if obs_ids[i] != obs_ids[i - 1] or event_ids[i] != event_ids[i - 1]:
            # append to subarray events
            multiplicities.append(multiplicity)
            subarray_obs_ids.append(obs_ids[i])
            subarray_event_ids.append(event_ids[i])
            subarray_first_index.append(i)
            # reset state
            current_idx += 1
            multiplicity = 0

        multiplicity += 1
        idx[i] = current_idx

    # Append multiplicity of the last subarray event
    if n_tel_events > 0:
        multiplicities.append(multiplicity)

    return SubarrayIndex(
        np.asarray(subarray_obs_ids),
        np.asarray(subarray_event_ids),
        np.asarray(multiplicities),
        np.asarray(subarray_first_index),
        idx,
    )


def get_subarray_index(tel_table) -> SubarrayIndex:
    """
    Get the subarray-event-wise information from a table of telescope events.

    Extract obs_ids and event_ids of all subarray events contained
    in a table of telescope events, their multiplicity and an array
    giving the index of the subarray event for each telescope event.

    This requires the telescope events of one subarray event to be come
    in a single block.

    Parameters
    ----------
    tel_table : astropy.table.Table
        table with telescope events as rows

    Returns
    -------
    subarray_index : SubarrayIndex
        obs_id of each subarray event,
        event_id of each subarray event,
        multiplicity (number of telescope events) for each subarray event,
        first index in the telescope event table for each subarray event,
        index into the subarray event array for each telescope event
    """
    obs_idx = tel_table["obs_id"]
    event_idx = tel_table["event_id"]
    return _get_subarray_index(obs_idx, event_idx)


def _grouped_add(tel_data, n_array_events, indices):
    """
    Calculate the group-wise sum for each array event over the
    corresponding telescope events. ``indices`` is an array
    that gives the index of the subarray event for each telescope event.
    """
    combined_values = np.zeros(n_array_events)
    np.add.at(combined_values, indices, tel_data)
    return combined_values


def weighted_mean_std_ufunc(
    tel_values,
    valid_tel,
    indices,
    multiplicity,
    weights=np.array([1]),
):
    """
    Calculate the weighted mean and standard deviation for each array event
    over the corresponding telescope events.

    Parameters
    ----------
    tel_values: np.ndarray
        values for each telescope event
    valid_tel: array-like
        boolean mask giving the valid values of ``tel_values``
    indices: np.ndarray
        index of the subarray event for each telescope event, returned as
        the fourth return value of ``get_subarray_index``
    multiplicity: np.ndarray
        multiplicity of the subarray events in the same order as the order of
        subarray events in ``indices``
    weights: np.ndarray
        weights used for averaging (equal/no weights are used by default)

    Returns
    -------
    Tuple(np.ndarray, np.ndarray)
        weighted mean and standard deviation for each array event
    """
    n_array_events = len(multiplicity)
    # avoid numerical problems by very large or small weights
    weights = weights / weights.max()
    tel_values = tel_values[valid_tel]
    indices = indices[valid_tel]

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

    sum_sq_residulas = _grouped_add(
        (tel_values - np.repeat(mean, multiplicity)[valid_tel]) ** 2 * weights,
        n_array_events,
        indices,
    )
    variance = np.full(n_array_events, np.nan)
    variance[valid] = sum_sq_residulas[valid] / sum_of_weights[valid]
    return mean, np.sqrt(variance)
