"""Helper functions for vectorizing numpy operations."""
import numpy as np
from numba import njit, uint64

__all__ = ["get_subarray_index", "weighted_mean_std_ufunc", "max_ufunc", "min_ufunc"]


@njit
def _get_subarray_index(obs_ids, event_ids):
    n_tel_events = len(obs_ids)
    idx = np.zeros(n_tel_events, dtype=uint64)
    current_idx = 0
    subarray_obs_index = []
    subarray_event_index = []
    multiplicities = []
    multiplicity = 0

    if n_tel_events > 0:
        subarray_obs_index.append(obs_ids[0])
        subarray_event_index.append(event_ids[0])
        multiplicity += 1

    for i in range(1, n_tel_events):
        if obs_ids[i] != obs_ids[i - 1] or event_ids[i] != event_ids[i - 1]:
            # append to subarray events
            multiplicities.append(multiplicity)
            subarray_obs_index.append(obs_ids[i])
            subarray_event_index.append(event_ids[i])
            # reset state
            current_idx += 1
            multiplicity = 0

        multiplicity += 1
        idx[i] = current_idx

    # Append multiplicity of the last subarray event
    if n_tel_events > 0:
        multiplicities.append(multiplicity)

    return (
        np.asarray(subarray_obs_index),
        np.asarray(subarray_event_index),
        np.asarray(multiplicities),
        idx,
    )


def get_subarray_index(tel_table):
    """
    Get the obs_ids and event_ids of all subarray events contained
    in a table of telescope events, their multiplicity and an array
    giving the index of the subarray event for each telescope event.
    This requires the telescope events to be SORTED by their corresponding
    subarray events (meaning by ``["obs_id", "event_id"]``).

    Parameters
    ----------
    tel_table: astropy.table.Table
        table with telescope events as rows

    Returns
    -------
    Tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        obs_ids of subarray events, event_ids of subarray events,
        multiplicity of subarray events, index of the subarray event
        for each telescope event
    """
    obs_idx = tel_table["obs_id"]
    event_idx = tel_table["event_id"]
    return _get_subarray_index(obs_idx, event_idx)


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


def weighted_mean_std_ufunc(
    tel_values,
    valid_tel,
    n_array_events,
    indices,
    multiplicity,
    weights=np.array([1]),
):
    """
    Calculate the weighted mean and standart deviation for each array event
    over the corresponding telescope events.

    Parameters
    ----------
    tel_values: np.ndarray
        values for each telescope event
    valid_tel: array-like
        boolean mask giving the valid values of ``tel_values``
    n_array_events: int
        number of array events with corresponding telescope events in ``tel_values``
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
        weighted mean and standart deviation for each array event
    """
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


def max_ufunc(tel_values, valid_tel, n_array_events, indices):
    """
    Find the maximum value for each array event from the
    corresponding telescope events.

    Parameters
    ----------
    tel_values: np.ndarray
        values for each telescope event
    valid_tel: array-like
        boolean mask giving the valid values of ``tel_values``
    n_array_events: int
        number of array events with corresponding telescope events in ``tel_values``
    indices: np.ndarray
        index of the subarray event for each telescope event, returned as
        the fourth return value of ``get_subarray_index``

    Returns
    -------
    np.ndarray
        maximum value for each array event
    """
    max_values = np.full(n_array_events, -np.inf)
    np.maximum.at(max_values, indices[valid_tel], tel_values[valid_tel])

    result = np.full(n_array_events, np.nan)
    valid = max_values > -np.inf
    result[valid] = max_values[valid]
    return result


def min_ufunc(tel_values, valid_tel, n_array_events, indices):
    """
    Find the minimum value for each array event from the
    corresponding telescope events.

    Parameters
    ----------
    tel_values: np.ndarray
        values for each telescope event
    valid_tel: array-like
        boolean mask giving the valid values of ``tel_values``
    n_array_events: int
        number of array events with corresponding telescope events in ``tel_values``
    indices: np.ndarray
        index of the subarray event for each telescope event, returned as
        the fourth return value of ``get_subarray_index``

    Returns
    -------
    np.ndarray
        minimum value for each array event
    """
    min_values = np.full(n_array_events, np.inf)
    np.minimum.at(min_values, indices[valid_tel], tel_values[valid_tel])

    result = np.full(n_array_events, np.nan)
    valid = min_values < np.inf
    result[valid] = min_values[valid]
    return result
