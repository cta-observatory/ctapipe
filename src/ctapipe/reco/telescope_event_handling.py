"""Helper functions for array-event-wise aggregation of telescope events."""

from functools import lru_cache
from itertools import combinations

import astropy.units as u
import numpy as np
from numba import njit, uint64

__all__ = [
    "get_subarray_index",
    "weighted_mean_std_ufunc",
    "get_combinations",
    "calc_combs_min_distances_event",
    "calc_combs_min_distances_table",
    "calc_fov_lon_lat",
    "create_combs_array",
    "get_index_combs",
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
    Compute the sum of telescope event data for each corresponding array event.

    It groups telescope event values by their corresponding
    array event indices and computes the sum for each group.

    Parameters
    ----------
    tel_data : np.ndarray
        Array of telescope event data to be summed.
    n_array_events : int
        Total number of array events.
    indices : np.ndarray
        Index array mapping each telescope event to its corresponding array event.

    Returns
    -------
    np.ndarray
        Array of summed values for each array event.
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


@lru_cache(maxsize=4096)
def get_combinations(iterable, size):
    """
    Generate all possible combinations of elements of a given `size` from
    the given `iterable`.

    Uses ``itertools.combinations`` and caching to speed up repeated calls.

    Parameters
    ----------
    iterable: iterable
        Input iterable from which to generate combinations.
    size : int
        The size of each combination.

    Returns
    -------
    np.ndarray
        Array of combinations of the specified size.
    """
    return np.array(list(combinations(iterable, size)))


@njit
def calc_combs_min_distances_event(
    index_tel_combs, fov_lon_values, fov_lat_values, weights, dist_weights
):
    """
    Calculate the weighted mean field-of-view (FoV) coordinates for each telescope combination.

    Determines the weighted minimum distance between all possible telescopes SIGN
    pairs per telescope combination and computes their weighted mean FoV longitude and latitude.
    Used event-wise with njit decorator.

    Parameters
    ----------
    index_tel_combs : np.ndarray
        Array of shape (n_combs, 2) containing index pairs of the valid telescope IDs per
        array event forming combinations.
    fov_lon_values : np.ndarray
        Array of shape (n_tels, 2) containing the field-of-view longitude values for each
        SIGN value (-1, 1) per telescope event.
    fov_lat_values : np.ndarray
        Array of shape (n_tels, 2) containing the field-of-view latitude values for each
        SIGN value (-1, 1) per telescope event.
    weights : np.ndarray
        Array of weights for each telescope event.
    dist_weights : np.ndarray
        Array of shape (n_tels, 2) containing adapted SIGN scores for each sign per
        telescope event.

    Returns
    -------
    Tuple(np.ndarray, np.ndarray, np.ndarray)
        - Weighted mean FoV longitude values for each combination with the minimum distance
          SIGN pair.
        - Weighted mean FoV latitude values for each combination with the minimum distance
          SIGN pair.
        - Combined weights of each telescope combination.
    """
    n_combs = len(index_tel_combs)

    combined_weights = np.empty(n_combs, dtype=np.float64)
    fov_lons = np.empty(n_combs, dtype=np.float64)
    fov_lats = np.empty(n_combs, dtype=np.float64)

    sign_combs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    for i in range(n_combs):
        tel_1, tel_2 = index_tel_combs[i]

        # Calculate weights
        w1, w2 = weights[tel_1], weights[tel_2]
        combined_weights[i] = w1 + w2

        # Calculate scores for each telescope combination using the
        # dist weights (SIGN scores). If one of the four weights of a telescope
        # combination is != 1, set the others to inf to avoid selecting them.
        comb_dist_weights = (
            dist_weights[tel_1, sign_combs[:, 0]]
            * dist_weights[tel_2, sign_combs[:, 1]]
        )
        if np.any(comb_dist_weights != 1):
            comb_dist_weights = np.where(
                comb_dist_weights == 1, np.inf, comb_dist_weights
            )

        # Calculate all 4 possible distances and weight them
        lon_diffs = (
            fov_lon_values[tel_1, sign_combs[:, 0]]
            - fov_lon_values[tel_2, sign_combs[:, 1]]
        )
        lat_diffs = (
            fov_lat_values[tel_1, sign_combs[:, 0]]
            - fov_lat_values[tel_2, sign_combs[:, 1]]
        )

        distances = np.hypot(lon_diffs, lat_diffs) * comb_dist_weights**2
        argmin_distance = np.argmin(distances)

        # Weighted mean for minimum distances
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


def calc_combs_min_distances_table(
    index_tel_combs,
    fov_lon_values,
    fov_lat_values,
    weights,
    dist_weights,
):
    """
    Calculate the weighted mean field-of-view (FoV) coordinates for each telescope combination.

    Determines the minimum distance between all possible telescopes SIGN
    pairs per telescope combination and computes their weighted mean FoV longitude and latitude.
    Used for table-wise broadcasting.

    Parameters
    ----------
    index_tel_combs : np.ndarray
        Array of shape (n_combs, 2) containing index pairs of the valid telescope IDs per
        array event forming combinations.
    fov_lon_values : np.ndarray
        Array of shape (n_tels, 2) containing the field-of-view longitude values for each
        SIGN value (-1, 1) per telescope event.
    fov_lat_values : np.ndarray
        Array of shape (n_tels, 2) containing the field-of-view latitude values for each
        SIGN value (-1, 1) per telescope event.
    weights : np.ndarray
        Array of weights for each telescope event.
    dist_weights : np.ndarray
        Array of shape (n_tels, 2) containing adapted SIGN scores for each sign per
        telescope event.

    Returns
    -------
    Tuple(np.ndarray, np.ndarray, np.ndarray)
        - Weighted mean FoV longitude values for each combination with the minimum distance
          SIGN pair.
        - Weighted mean FoV latitude values for each combination with the minimum distance
          SIGN pair.
        - Combined weights for each telescope combination.
    """
    # Adding weights for each telescope combination
    mapped_weights = weights[index_tel_combs]
    combined_weights = np.add(mapped_weights[:, 0], mapped_weights[:, 1])

    sign_combs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    lon_combs = fov_lon_values[index_tel_combs]
    lon_diffs = lon_combs[:, 0, sign_combs[:, 0]] - lon_combs[:, 1, sign_combs[:, 1]]

    lat_combs = fov_lat_values[index_tel_combs]
    lat_diffs = lat_combs[:, 0, sign_combs[:, 0]] - lat_combs[:, 1, sign_combs[:, 1]]

    # Calculate scores for each telescope combination using the
    # dist weights (SIGN scores). If one of the four weights of a telescope
    # combination is != 1, set the others to inf to avoid selecting them.
    # Calculate all 4 possible distances afterwards.
    comb_dist_weights = (
        dist_weights[index_tel_combs][:, 0, sign_combs[:, 0]]
        * dist_weights[index_tel_combs][:, 1, sign_combs[:, 1]]
    )

    dist_weight_mask = np.any(comb_dist_weights != 1, axis=1)
    if np.any(dist_weight_mask):
        comb_dist_weights[dist_weight_mask] = np.where(
            comb_dist_weights[dist_weight_mask] == 1,
            np.inf,
            comb_dist_weights[dist_weight_mask],
        )
    distances = np.hypot(lon_diffs, lat_diffs) * comb_dist_weights**2
    argmin_distance = np.argmin(distances, axis=1)

    # Weighted mean for minimum distances
    lon_values = np.array(
        [
            fov_lon_values[index_tel_combs[:, 0], sign_combs[argmin_distance, 0]],
            fov_lon_values[index_tel_combs[:, 1], sign_combs[argmin_distance, 1]],
        ]
    )

    lat_values = np.array(
        [
            fov_lat_values[index_tel_combs[:, 0], sign_combs[argmin_distance, 0]],
            fov_lat_values[index_tel_combs[:, 1], sign_combs[argmin_distance, 1]],
        ]
    )

    weighted_lons = np.average(lon_values, weights=mapped_weights.T, axis=0)
    weighted_lats = np.average(lat_values, weights=mapped_weights.T, axis=0)

    return weighted_lons, weighted_lats, combined_weights


def calc_fov_lon_lat(tel_table, sign_score_limit, prefix="DispReconstructor_tel"):
    """
    Calculate possible field-of-view (FoV) longitude and latitude coordinates
    and weights.

    For each telescope event, this function computes the two possible
    (fov_lon, fov_lat) positions in the telescope frame corresponding to the
    DISP direction ambiguity (SIGN = ±1). Moreover, it calculates weights based
    on the DISP sign score used for calculating the minimum distance per telescope
    combination later on.

    Parameters
    ----------
    tel_table : astropy.table.Table
        Table containing Hillas parameters and DISP reconstruction results.
        Must include the following columns:
        - ``hillas_fov_lon`` : Hillas ellipse centroid longitude (Quantity)
        - ``hillas_fov_lat`` : Hillas ellipse centroid latitude (Quantity)
        - ``hillas_psi`` : Hillas ellipse orientation angle (Quantity)
        - ``<prefix>_parameter`` : DISP distance from image centroid (float)
        - ``<prefix>_sign_score`` : DISP sign score (float) if ``sign_score_limit``
          is not None.
    sign_score_limit : float or None
        Minimum DISP sign score to consider when calculating the dist weights
        (1 / (1 + sign_score)). Weights of events with
        ``sign_score < sign_score_limit`` and not selected DISP signs are
        set to 1.
    prefix : str, optional
        Prefix used to access the DISP parameter (default: "DispReconstructor_tel").

    Returns
    -------
    tuple of np.ndarray
        (fov_lon, fov_lat, disp_scores)

        - ``fov_lon`` : array of shape (n_tel_events, 2)
          Possible FoV longitudes for each sign (-1, +1) in degrees.
        - ``fov_lat`` : array of shape (n_tel_events, 2)
          Possible FoV latitudes for each sign (-1, +1) in degrees.
        - ``dist_weights`` : array of shape (n_tel_events, 2)
          Weights (1 / (1 + sign_score)) for each telescope sign. 1 for
          sign_scores below the given ``sign_score_limit`` and not selected
          DISP signs. Between 0 and 1. Lower values correspond to more reliable
          reconstruction.
    """
    hillas_fov_lon = tel_table["hillas_fov_lon"].quantity.to_value(u.deg)
    hillas_fov_lat = tel_table["hillas_fov_lat"].quantity.to_value(u.deg)
    hillas_psi = tel_table["hillas_psi"].quantity.to_value(u.rad)
    disp = tel_table[f"{prefix}_parameter"]
    signs = np.array([-1, 1])

    dist_weights = np.ones((len(disp), 2))
    if sign_score_limit is not None:
        sign_score = tel_table[f"{prefix}_sign_score"]
        mask_sign = np.sign(disp)[:, None] == signs
        sign_score[sign_score <= sign_score_limit] = 0
        dist_weights[mask_sign] = 1 / (1 + sign_score)

    cos_psi = np.cos(hillas_psi)
    sin_psi = np.sin(hillas_psi)
    lons = hillas_fov_lon[:, None] + signs * np.abs(disp)[:, None] * cos_psi[:, None]
    lats = hillas_fov_lat[:, None] + signs * np.abs(disp)[:, None] * sin_psi[:, None]

    return lons, lats, dist_weights


def create_combs_array(max_multiplicity, k):
    """
    Generate an array of all possible `k`-combinations for multiplicities up to
    `max_multiplicity`.

    Precomputes and stores combinations for different multiplicities to reach them
    by index afterwards.

    Parameters
    ----------
    max_multiplicity : int
        Maximum multiplicity to consider.
    k : int
        The size of the combinations.

    Returns
    -------
    Tuple(np.ndarray, np.ndarray)
        - An array of all k-combinations for different multiplicities.
        - An array mapping each combination to its respective multiplicity.
    """
    combs_array = get_combinations(range(k), k)
    for i in range(k + 1, max_multiplicity + 1):
        combs = get_combinations(range(i), k)
        combs_array = np.concatenate([combs_array, combs])

    n_combs = _calc_n_combs(np.arange(k, max_multiplicity + 1), k)
    combs_to_multi_indices = np.repeat(np.arange(k, max_multiplicity + 1), n_combs)

    return combs_array, combs_to_multi_indices


@njit
def _binomial(n, k):
    """
    Compute the binomial coefficient (`n` choose `k`).

    Parameters
    ----------
    n : int
        Total number of items.
    k : int
        Number of selections.

    Returns
    -------
    int
        The binomial coefficient (n choose k).
    """
    if k > n or k < 0:
        return 0
    k = min(k, n - k)
    c = 1
    for i in range(k):
        c = c * (n - i) // (i + 1)
    return c


@njit
def _calc_n_combs(multiplicity, k):
    """
    Calculate the number of possible `k`-combinations for each `multiplicity` value.

    Parameters
    ----------
    multiplicity : np.ndarray
        Array of multiplicity values for each array event.
    k : int
        The size of the combinations.

    Returns
    -------
    np.ndarray
        Array of combination counts corresponding to each multiplicity.
    """
    n_combs = np.empty(len(multiplicity), dtype=np.int64)
    for i in range(len(multiplicity)):
        n_combs[i] = _binomial(multiplicity[i], k)

    return n_combs


@njit
def get_index_combs(multiplicities, combs_array, combs_to_multi_indices, k):
    """
    Generate the telescope event indices for all `k`-combinations of telescope events based on
    `multiplicities`.

    Returns also an array containing the number of combinations per multiplicity.

    Parameters
    ----------
    multiplicities : np.ndarray
        Array of multiplicity values for each array event.
    combs_array : np.ndarray
        Precomputed combinations for different multiplicities.
    combs_to_multi_indices : np.ndarray
        Array mapping combinations to corresponding multiplicities.
    k : int
        The size of the combinations.

    Returns
    -------
    Tuple(np.ndarray, np.ndarray)
        - Array of index combinations for telescope events.
        - Array containing the number of combinations per multiplicity.
    """
    n_combs = _calc_n_combs(multiplicities, k)
    total_combs = np.sum(n_combs)
    index_tel_combs = np.empty((total_combs, k), dtype=np.int64)
    cum_multiplicities = 0
    idx = 0

    for i in range(len(multiplicities)):
        mask = combs_to_multi_indices == multiplicities[i]
        selected_combs = combs_array[mask] + cum_multiplicities
        index_tel_combs[idx : idx + len(selected_combs)] = selected_combs
        idx += len(selected_combs)
        cum_multiplicities += multiplicities[i]

    return index_tel_combs, n_combs
