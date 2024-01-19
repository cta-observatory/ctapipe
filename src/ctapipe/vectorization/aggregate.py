import numpy as np

__all__ = ["weighted_mean_ufunc", "max_ufunc", "min_ufunc"]


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


def weighted_mean_ufunc(tel_values, weights, n_array_events, indices):
    """
    Calculate the weighted mean for each array event over the
    corresponding telescope events.

    Parameters
    ----------
    tel_values: np.ndarray
        values for each telescope event
    weights: np.ndarray
        weights used for averaging
    n_array_events: int
        number of array events with corresponding telescope events in ``tel_values``
    indices: np.ndarray
        index of the subarray event for each telescope event, returned by
        ``np.unique(tel_events[["obs_id", "event_id"]], return_inverse=True)``

    Returns
    -------
    array: np.ndarray
        weighted mean for each array event
    """
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


def max_ufunc(tel_values, n_array_events, indices):
    """
    Find the maximum value for each array event from the
    corresponding telescope events.

    Parameters
    ----------
    tel_values: np.ndarray
        values for each telescope event
    n_array_events: int
        number of array events with corresponding telescope events in ``tel_values``
    indices: np.ndarray
        index of the subarray event for each telescope event, returned by
        ``np.unique(tel_events[["obs_id", "event_id"]], return_inverse=True)``

    Returns
    -------
    array: np.ndarray
        maximum value for each array event
    """
    if np.issubdtype(tel_values[0], np.integer):
        fillvalue = np.iinfo(tel_values.dtype).min
    elif np.issubdtype(tel_values[0], np.floating):
        fillvalue = np.finfo(tel_values.dtype).min
    else:
        raise ValueError("Non-numerical dtypes are not supported")

    max_values = np.full(n_array_events, fillvalue)
    np.maximum.at(max_values, indices, tel_values)

    result = np.full(n_array_events, np.nan)
    valid = max_values > fillvalue
    result[valid] = max_values[valid]
    return result


def min_ufunc(tel_values, n_array_events, indices):
    """
    Find the minimum value for each array event from the
    corresponding telescope events.

    Parameters
    ----------
    tel_values: np.ndarray
        values for each telescope event
    n_array_events: int
        number of array events with corresponding telescope events in ``tel_values``
    indices: np.ndarray
        index of the subarray event for each telescope event, returned by
        ``np.unique(tel_events[["obs_id", "event_id"]], return_inverse=True)``

    Returns
    -------
    array: np.ndarray
        minimum value for each array event
    """
    if np.issubdtype(tel_values[0], np.integer):
        fillvalue = np.iinfo(tel_values.dtype).max
    elif np.issubdtype(tel_values[0], np.floating):
        fillvalue = np.finfo(tel_values.dtype).max
    else:
        raise ValueError("Non-numerical dtypes are not supported")

    min_values = np.full(n_array_events, fillvalue)
    np.minimum.at(min_values, indices, tel_values)

    result = np.full(n_array_events, np.nan)
    valid = min_values < fillvalue
    result[valid] = min_values[valid]
    return result
