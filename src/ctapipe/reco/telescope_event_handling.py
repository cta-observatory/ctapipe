"""Helper functions for array-event-wise aggregation of telescope events."""

from functools import lru_cache
from itertools import combinations

import astropy.units as u
import numpy as np
from numba import njit, uint64

from ctapipe.image.statistics import argmin

__all__ = [
    "get_subarray_index",
    "weighted_mean_std_ufunc",
    "get_combinations",
    "calc_combs_min_distances",
]


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
    Get the subarray-event-wise information from a table of telescope events.

    Extract obs_ids and event_ids of all subarray events contained
    in a table of telescope events, their multiplicity and an array
    giving the index of the subarray event for each telescope event.

    This requires the telescope events of one subarray event to be
    in a single block.

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

    sum_sq_residuals = _grouped_add(
        (tel_values - np.repeat(mean, multiplicity)[valid_tel]) ** 2 * weights,
        n_array_events,
        indices,
    )
    variance = np.full(n_array_events, np.nan)
    variance[valid] = sum_sq_residuals[valid] / sum_of_weights[valid]
    return mean, np.sqrt(variance)


@njit
def calc_combs_min_distances(
    index_combs_tel_ids, fov_lon_values, fov_lat_values, weights
):
    """
    Returns the weighted average fov lon/lat for every telescope combination
    in tel_combs additional to the sum of their weights.
    """
    num_combs = len(index_combs_tel_ids)

    combined_weights = np.empty(num_combs, dtype=np.float64)
    fov_lons = np.empty(num_combs, dtype=np.float64)
    fov_lats = np.empty(num_combs, dtype=np.float64)

    sign_combs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    for i in range(num_combs):
        tel_1, tel_2 = index_combs_tel_ids[i]

        # Calculate weights
        w1, w2 = weights[tel_1], weights[tel_2]
        combined_weights[i] = w1 + w2

        # Calculate all 4 possible distances
        lon_diffs = (
            fov_lon_values[tel_1, sign_combs[:, 0]]
            - fov_lon_values[tel_2, sign_combs[:, 1]]
        )
        lat_diffs = (
            fov_lat_values[tel_1, sign_combs[:, 0]]
            - fov_lat_values[tel_2, sign_combs[:, 1]]
        )

        distances = np.hypot(lon_diffs, lat_diffs)

        # Weighted mean for minimum distances
        argmin_distance = argmin(distances)
        lon_vals = [
            fov_lon_values[tel_1, sign_combs[argmin_distance, 0]],
            fov_lon_values[tel_2, sign_combs[argmin_distance, 1]],
        ]

        lat_vals = [
            fov_lat_values[tel_1, sign_combs[argmin_distance, 0]],
            fov_lat_values[tel_2, sign_combs[argmin_distance, 1]],
        ]

        fov_lons[i] = np.average(lon_vals, weights=[w1, w2])
        fov_lats[i] = np.average(lat_vals, weights=[w1, w2])

    return fov_lons, fov_lats, combined_weights


@lru_cache(maxsize=4096)
def get_combinations(array, size):
    """
    Return all combinations of elements of a given size from an array.
    """
    return np.array(list(combinations(array, size)))


@njit
def binomial(n, k):
    if k > n or k < 0:
        return 0
    k = min(k, n - k)
    c = 1
    for i in range(k):
        c = c * (n - i) // (i + 1)
    return c


@njit
def calc_num_combs(multiplicity, k):
    num_combs = np.empty(len(multiplicity), dtype=np.int64)
    for i in range(len(multiplicity)):
        num_combs[i] = binomial(multiplicity[i], k)

    return num_combs


def calc_fov_lon_lat(valid_mono_predictions, prefix):
    hillas_fov_lon = valid_mono_predictions["hillas_fov_lon"].quantity.to_value(u.deg)
    hillas_fov_lat = valid_mono_predictions["hillas_fov_lat"].quantity.to_value(u.deg)
    hillas_psi = valid_mono_predictions["hillas_psi"].quantity.to_value(u.rad)
    disp = valid_mono_predictions[f"{prefix}_parameter"]
    signs = np.array([-1, 1])

    cos_psi = np.cos(hillas_psi)
    sin_psi = np.sin(hillas_psi)
    lons = hillas_fov_lon[:, None] + signs * disp[:, None] * cos_psi[:, None]
    lats = hillas_fov_lat[:, None] + signs * disp[:, None] * sin_psi[:, None]

    return lons, lats


def create_combs_array(max_multi, k):
    combs_map = get_combinations(range(2), k)
    for i in range(3, max_multi + 1):
        combs = get_combinations(range(i), k)
        combs_map = np.concatenate([combs_map, combs])

    num_combs = calc_num_combs(np.arange(2, max_multi + 1), k)
    combs_to_multi_indices = np.repeat(np.arange(2, max_multi + 1), num_combs)

    return combs_map, combs_to_multi_indices


@njit
def get_index_combs(multiplicity, combs_map, combs_to_multi_indices, k):
    num_combs = calc_num_combs(multiplicity, k)
    total_combs = np.sum(num_combs)
    index_tel_combs_map = np.empty((total_combs, k), dtype=np.int64)
    cum_multiplicity = 0
    idx = 0

    for i in range(len(multiplicity)):
        mask = combs_to_multi_indices == multiplicity[i]
        selected_combs = combs_map[mask] + cum_multiplicity
        index_tel_combs_map[idx : idx + len(selected_combs)] = selected_combs
        idx += len(selected_combs)
        cum_multiplicity += multiplicity[i]

    return index_tel_combs_map, num_combs


def calc_combs_min_distances_table(
    index_combs_tel_ids,
    fov_lon_values,
    fov_lat_values,
    weights,
):
    """
    Returns the weighted average fov lon/lat for every telescope combination
    in tel_combs additional to the sum of their weights.
    """
    # Adding weights for each telescope combination
    mapped_weights = weights[index_combs_tel_ids]
    combined_weights = np.add(mapped_weights[:, 0], mapped_weights[:, 1])

    sign_combs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # Calculate all 4 possible distances
    lon_combs = fov_lon_values[index_combs_tel_ids]
    lon_diffs = lon_combs[:, 0, sign_combs[:, 0]] - lon_combs[:, 1, sign_combs[:, 1]]

    lat_combs = fov_lat_values[index_combs_tel_ids]
    lat_diffs = lat_combs[:, 0, sign_combs[:, 0]] - lat_combs[:, 1, sign_combs[:, 1]]

    distances = np.hypot(lon_diffs, lat_diffs)
    argmin_distance = np.argmin(distances, axis=1)

    # Weighted mean for minimum distances
    lon_values = np.array(
        [
            fov_lon_values[index_combs_tel_ids[:, 0], sign_combs[argmin_distance, 0]],
            fov_lon_values[index_combs_tel_ids[:, 1], sign_combs[argmin_distance, 1]],
        ]
    )

    lat_values = np.array(
        [
            fov_lat_values[index_combs_tel_ids[:, 0], sign_combs[argmin_distance, 0]],
            fov_lat_values[index_combs_tel_ids[:, 1], sign_combs[argmin_distance, 1]],
        ]
    )

    weighted_lons = np.average(lon_values, weights=mapped_weights.T, axis=0)
    weighted_lats = np.average(lat_values, weights=mapped_weights.T, axis=0)

    return weighted_lons, weighted_lats, combined_weights


@njit
def calc_fov_lon_lat_njit(hillas_fov_lon, hillas_fov_lat, hillas_psi, disp):
    signs = np.array([-1, 1])

    cos_psi = np.cos(hillas_psi)
    sin_psi = np.sin(hillas_psi)
    lons = hillas_fov_lon[:, None] + signs * disp[:, None] * cos_psi[:, None]
    lats = hillas_fov_lat[:, None] + signs * disp[:, None] * sin_psi[:, None]

    return lons, lats


@njit
def calc_fov_njit(
    obs_ids,
    event_ids,
    valid_tels,
    valid_weights,
    hillas_fov_lon,
    hillas_fov_lat,
    hillas_psi,
    disp,
    combs_map,
    combs_to_multi_indices,
):
    n_tel_events = len(obs_ids)
    idx_min_list = []
    idx_min = 0
    subarray_counter = 0
    single_tel_mask = np.empty(n_tel_events, dtype=np.bool)
    single_tel_mask_array = []
    fov_lon_weighted_average = []
    fov_lat_weighted_average = []
    for i in range(1, n_tel_events):
        if (
            obs_ids[i] != obs_ids[i - 1]
            or event_ids[i] != event_ids[i - 1]
            or i == (n_tel_events - 1)
        ):
            valid_mask = valid_tels[idx_min:i]
            valid_multiplicity = valid_mask.sum()
            if valid_multiplicity >= 1:
                if valid_multiplicity == 1:
                    single_tel_mask[idx_min:i] = valid_mask
                    single_tel_mask_array.append(subarray_counter)
                    fov_lon_weighted_average.append(np.nan)
                    fov_lat_weighted_average.append(np.nan)
                else:
                    single_tel_mask[idx_min:i] = np.full(len(valid_mask), False)
                    fov_lon_values, fov_lat_values = calc_fov_lon_lat_njit(
                        hillas_fov_lon[idx_min:i][valid_mask],
                        hillas_fov_lat[idx_min:i][valid_mask],
                        hillas_psi[idx_min:i][valid_mask],
                        disp[idx_min:i][valid_mask],
                    )
                    index_combs_tel_ids = combs_map[
                        combs_to_multi_indices == valid_multiplicity
                    ]
                    (
                        min_dist_lon,
                        min_dist_lat,
                        combined_weights,
                    ) = calc_combs_min_distances(
                        index_combs_tel_ids,
                        fov_lon_values,
                        fov_lat_values,
                        valid_weights[idx_min:i],
                    )
                    fov_lon_weighted_average.append(
                        np.average(min_dist_lon, weights=combined_weights)
                    )
                    fov_lat_weighted_average.append(
                        np.average(min_dist_lat, weights=combined_weights)
                    )
            else:
                single_tel_mask[idx_min:i] = np.full(len(valid_mask), False)
                fov_lon_weighted_average.append(np.nan)
                fov_lat_weighted_average.append(np.nan)

            idx_min_list.append(idx_min)
            idx_min = i
            subarray_counter += 1

    return (
        fov_lon_weighted_average,
        fov_lat_weighted_average,
        single_tel_mask,
        single_tel_mask_array,
        idx_min_list,
    )
