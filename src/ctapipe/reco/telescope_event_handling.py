"""Helper functions for array-event-wise aggregation of telescope events."""

from functools import lru_cache
from itertools import combinations, product

import astropy.units as u
import numpy as np
from numba import njit, uint64

from ..core.env import CTAPIPE_DISABLE_NUMBA_CACHE

__all__ = [
    "get_subarray_index",
    "weighted_mean_std_ufunc",
    "get_combinations",
    "binary_combinations",
    "check_ang_diff",
    "calc_combs_min_distances",
    "valid_tels_of_multi",
    "fill_lower_multiplicities",
    "calc_fov_lon_lat",
    "create_combs_array",
    "get_index_combs",
]


@njit(cache=not CTAPIPE_DISABLE_NUMBA_CACHE)
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
def get_combinations(array_length: int, comb_size: int) -> np.ndarray:
    """
    Generate all possible index combinations of elements of a given `comb_size`
    from an array with a given `array_length`.

    Uses ``itertools.combinations`` and caching to speed up repeated calls.

    Parameters
    ----------
    array_length: int
        Length of the list or array to generate index combinations from.
    comb_size : int
        The size of each combination.

    Returns
    -------
    np.ndarray
        Array of index combinations of the specified size.
    """
    return np.array(list(combinations(range(array_length), comb_size)))


@lru_cache(maxsize=4096)
def binary_combinations(k: int) -> np.ndarray:
    """
    Generate all binary (0/1) combinations of length k.

    Uses ``itertools.product`` and caching to speed up repeated calls.

    Parameters
    ----------
    k : int
        Length of each binary combination.

    Returns
    -------
    np.ndarray
        A NumPy array of shape (2**k, k) containing all possible binary
        combinations, where each row represents one combination.
    """
    return np.array(list(product([0, 1], repeat=k)), dtype=int)


def check_ang_diff(min_ang_diff, psi1, psi2):
    """
    Check whether the angular separation between two reconstructed main shower
    axes is larger than a configured minimum value.

    This function compares the orientation angles of two Hillas main axes and
    computes an axis-invariant angular difference, where angles differing by
    180° are treated as parallel. It is therefore suitable for resolving the
    near-parallel-axis ambiguity in stereo reconstruction.

    The method supports both scalar (event-wise) and array-like (table-wise)
    inputs, enabling vectorized application to multiple two-telescope events.

    Parameters
    ----------
    psi1, psi2 : astropy.units.Quantity
        Orientation angles of the Hillas main shower axes. The inputs may be
        scalars or array-like objects of equal shape and must carry angular
        units (e.g. degrees).

    Returns
    -------
    bool or numpy.ndarray
        Boolean value or boolean array indicating whether the angular
        separation is greater than or equal to ``self.min_ang_diff``.
        ``True`` means the axes are sufficiently separated and the event
        should be kept; ``False`` means the axes are too parallel and the
        event should be rejected.

    Notes
    -----
    - The angular separation is folded into the range [0°, 90°], such that
      axis orientations differing by 0° or 180° are treated equivalently.
    - If used in a table-wise context, the comparison is fully vectorized
      and no Python-level loops are required.
    """
    ang_diff = np.abs(psi1 - psi2) % (180 * u.deg)
    ang_diff = np.minimum(ang_diff, 180 * u.deg - ang_diff)
    return ang_diff >= (min_ang_diff * u.deg)


def calc_combs_min_distances(index_tel_combs, fov_lon_values, fov_lat_values, weights):
    """
    Determine the optimal DISP sign combination for each telescope combination
    by minimizing the weighted sum of squared distances (SSE) to the
    weighted mean direction.

    For each telescope combination of size ``k``, all ``2**k`` possible
    sign assignments (corresponding to the DISP head–tail ambiguity) are
    evaluated. For every sign assignment, the weighted mean field-of-view
    (FoV) position is computed and the weighted sum of squared distances
    of the individual telescope directions to this mean (SSE) is calculated.
    The sign assignment that minimizes this SSE is selected.

    The function then returns the weighted mean FoV longitude and latitude
    for the best sign assignment of each telescope combination, together
    with the combined weight of the combination.

    Parameters
    ----------
    index_tel_combs : np.ndarray
        Array of shape ``(n_combs, k)`` containing index combinations of
        telescope events forming each combination.
    fov_lon_values : np.ndarray
        Array of shape ``(n_tel, 2)`` containing the two possible FoV longitude
        values (SIGN = ±1) for each telescope event, in degrees.
    fov_lat_values : np.ndarray
        Array of shape ``(n_tel, 2)`` containing the two possible FoV latitude
        values (SIGN = ±1) for each telescope event, in degrees.
    weights : np.ndarray
        Array of shape ``(n_tel,)`` with the weight assigned to each
        telescope event.

    Returns
    -------
    weighted_lons : np.ndarray
        Array of shape ``(n_combs,)`` containing the weighted mean FoV
        longitudes of the best sign assignment for each telescope combination.
    weighted_lats : np.ndarray
        Array of shape ``(n_combs,)`` containing the weighted mean FoV
        latitudes of the best sign assignment for each telescope combination.
    combined_weights : np.ndarray
        Array of shape ``(n_combs,)`` containing the sum of weights of the
        telescopes contributing to each combination.

    """
    mapped_weights = weights[index_tel_combs]  # (n_combs, k)
    combined_weights = mapped_weights.sum(axis=1)

    _, k = index_tel_combs.shape
    sign_combs = binary_combinations(k)  # (2**k, k)

    lon_combs = fov_lon_values[index_tel_combs]  # (n_combs, k, 2)
    lat_combs = fov_lat_values[index_tel_combs]  # (n_combs, k, 2)

    sign_idx = sign_combs.T[None, :, :]  # (1, k, 2**k) broadcasted
    lon_vals = np.take_along_axis(lon_combs, sign_idx, axis=2)  # (n_combs, k, 2**k)
    lat_vals = np.take_along_axis(lat_combs, sign_idx, axis=2)  # (n_combs, k, 2**k)

    w = mapped_weights[:, :, None]  # (n_combs, k, 1)
    wsum = w.sum(axis=1)  # (n_combs, 1)

    lon_mu = (lon_vals * w).sum(axis=1) / wsum  # (n_combs, 2**k)
    lat_mu = (lat_vals * w).sum(axis=1) / wsum  # (n_combs, 2**k)

    # SSE
    dlon = lon_vals - lon_mu[:, None, :]
    dlat = lat_vals - lat_mu[:, None, :]
    sse = (w * (dlon * dlon + dlat * dlat)).sum(axis=1)  # (n_combs, 2**k)

    argmin = np.argmin(sse, axis=1)  # (n_combs,)
    best_signs = sign_combs[argmin]  # (n_combs, k)

    best_idx = best_signs[:, :, None]  # (n_combs, k, 1)
    best_lon = np.take_along_axis(lon_combs, best_idx, axis=2)[:, :, 0]  # (n_combs, k)
    best_lat = np.take_along_axis(lat_combs, best_idx, axis=2)[:, :, 0]  # (n_combs, k)

    weighted_lons = np.average(best_lon, weights=mapped_weights, axis=1)
    weighted_lats = np.average(best_lat, weights=mapped_weights, axis=1)

    return weighted_lons, weighted_lats, combined_weights


def valid_tels_of_multi(multi, valid_tel_to_array_indices):
    """
    Create a boolean mask selecting telescope-event rows that belong to
    array events with a given multiplicity.

    The function assumes that telescope events are grouped contiguously
    by array event, i.e. all rows belonging to the same array event appear
    in a single consecutive block in ``valid_tel_to_array_indices``.
    Under this assumption, the multiplicity of an array event corresponds
    to the length of its contiguous block.

    For all array events whose block length equals ``multi``, the function
    returns a boolean mask that is ``True`` for all telescope rows belonging
    to those events and ``False`` otherwise.

    This block-based approach avoids repeated membership tests (e.g.
    ``np.isin``) and is therefore typically faster for large arrays.

    Parameters
    ----------
    multi : int
        Target multiplicity (number of telescope events per array event).
    valid_tel_to_array_indices : np.ndarray
        One-dimensional array mapping each valid telescope event to its
        corresponding array-event index. Must be ordered such that
        telescope events of the same array event are contiguous.

    Returns
    -------
    mask : np.ndarray
        Boolean array of the same length as ``valid_tel_to_array_indices``.
        Entries are ``True`` for telescope-event rows belonging to array
        events with multiplicity ``multi`` and ``False`` otherwise.

    """
    change = np.empty(len(valid_tel_to_array_indices), dtype=bool)
    change[0] = True
    change[1:] = valid_tel_to_array_indices[1:] != valid_tel_to_array_indices[:-1]

    starts = np.flatnonzero(change)
    lengths = np.diff(np.append(starts, len(valid_tel_to_array_indices)))
    good_blocks = lengths == multi
    mask = np.repeat(good_blocks, lengths)

    return mask


def fill_lower_multiplicities(
    fov_lon_combs_mean,
    fov_lat_combs_mean,
    n_tel_combinations,
    valid_tel_to_array_indices,
    valid_multiplicity,
    fov_lon_values,
    fov_lat_values,
    weights,
):
    """
    Fill stereo FoV longitude and latitude estimates for array events with
    multiplicities smaller than ``n_tel_combinations``.

    For array events whose telescope multiplicity is lower than
    ``n_tel_combinations`` but at least two, this function recomputes the
    stereo direction using all available telescopes of the event
    (i.e. combinations of size equal to the event multiplicity).
    The optimal DISP sign assignment for each event is determined via
    :func:`calc_combs_min_distances`.

    The resulting weighted mean FoV longitude and latitude values are written
    in-place into ``fov_lon_combs_mean`` and ``fov_lat_combs_mean`` at the
    positions corresponding to the affected array events.

    Parameters
    ----------
    fov_lon_combs_mean : np.ndarray
        Array of shape ``(n_array_events,)`` holding the mean FoV longitudes
        per array event. Values for lower-multiplicity events are updated
        in-place.
    fov_lat_combs_mean : np.ndarray
        Array of shape ``(n_array_events,)`` holding the mean FoV latitudes
        per array event. Values for lower-multiplicity events are updated
        in-place.
    n_tel_combinations : int
        Nominal number of telescopes used per combination in the stereo
        reconstruction.
    valid_tel_to_array_indices : np.ndarray
        Array mapping each valid telescope event to its corresponding
        array-event index. Telescope events must be grouped contiguously
        by array event.
    valid_multiplicity : np.ndarray
        Array of telescope multiplicities for each valid array event.
    fov_lon_values : np.ndarray
        Array of shape ``(n_valid_tel_events, 2)`` containing the two possible
        FoV longitude values (SIGN = ±1) for each valid telescope event.
    fov_lat_values : np.ndarray
        Array of shape ``(n_valid_tel_events, 2)`` containing the two possible
        FoV latitude values (SIGN = ±1) for each valid telescope event.
    weights : np.ndarray
        Array of weights for each valid telescope event.

    Returns
    -------
    None
        The function operates in-place and does not return a value.

    Notes
    -----
    - Only multiplicities of two or larger are considered; single-telescope
      events are handled separately.
    - The correctness of this function relies on the telescope-event rows
      being grouped contiguously by array event.
    - Using a block-based mask via :func:`valid_tels_of_multi` avoids repeated
      membership tests and is typically faster than approaches based on
      ``np.isin`` for large tables.
    """
    for multi in range(n_tel_combinations - 1, 1, -1):
        multi_mask = valid_multiplicity == multi
        if not np.any(multi_mask):
            continue

        mask = valid_tels_of_multi(multi, valid_tel_to_array_indices)
        index_tel_combs = np.arange(len(valid_tel_to_array_indices))[mask].reshape(
            -1, multi
        )

        lons, lats, _ = calc_combs_min_distances(
            index_tel_combs,
            fov_lon_values,
            fov_lat_values,
            weights,
        )
        fov_lon_combs_mean[multi_mask] = lons
        fov_lat_combs_mean[multi_mask] = lats


def calc_fov_lon_lat(tel_table, prefix="DispReconstructor_tel"):
    """
    Compute possible field-of-view (FoV) longitude and latitude coordinates
    for each telescope event, accounting for the DISP sign ambiguity.

    For every telescope event, two candidate shower directions in the
    telescope FoV are calculated corresponding to the two possible DISP
    sign assignments (SIGN = −1 and +1). The directions are obtained by
    shifting the Hillas image centroid along the main shower axis by the
    absolute DISP distance.

    The resulting FoV longitude and latitude values are returned as
    two-element arrays per telescope event and are typically used as
    inputs to subsequent stereo-combination steps (e.g. via
    :func:`calc_combs_min_distances`).

    Parameters
    ----------
    tel_table : astropy.table.Table
        Table containing Hillas parameters and DISP reconstruction results.
        Must include the following columns:
        - ``hillas_fov_lon`` : Hillas ellipse centroid longitude (Quantity)
        - ``hillas_fov_lat`` : Hillas ellipse centroid latitude (Quantity)
        - ``hillas_psi`` : Hillas ellipse orientation angle (Quantity)
        - ``<prefix>_parameter`` : DISP distance from image centroid (float)
    prefix : str, optional
        Prefix used to access the DISP parameter (default: "DispReconstructor_tel").

    Returns
    -------
    fov_lon : np.ndarray
        Array of shape ``(n_tel_events, 2)`` containing the possible FoV
        longitude values (in degrees) for each telescope event, corresponding
        to DISP signs −1 and +1.
    fov_lat : np.ndarray
        Array of shape ``(n_tel_events, 2)`` containing the possible FoV
        latitude values (in degrees) for each telescope event, corresponding
        to DISP signs −1 and +1.

    """
    hillas_fov_lon = tel_table["hillas_fov_lon"].quantity.to_value(u.deg)
    hillas_fov_lat = tel_table["hillas_fov_lat"].quantity.to_value(u.deg)
    hillas_psi = tel_table["hillas_psi"].quantity.to_value(u.rad)
    disp = tel_table[f"{prefix}_parameter"]
    signs = np.array([-1, 1])

    cos_psi = np.cos(hillas_psi)
    sin_psi = np.sin(hillas_psi)
    abs_disp = np.abs(disp)[:, None]
    lons = hillas_fov_lon[:, None] + signs * abs_disp * cos_psi[:, None]
    lats = hillas_fov_lat[:, None] + signs * abs_disp * sin_psi[:, None]

    return lons, lats


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
    combs_array = get_combinations(k, k)
    for i in range(k + 1, max_multiplicity + 1):
        combs = get_combinations(i, k)
        combs_array = np.concatenate([combs_array, combs])

    n_combs = _calc_n_combs(np.arange(k, max_multiplicity + 1), k)
    combs_to_multi_indices = np.repeat(np.arange(k, max_multiplicity + 1), n_combs)

    return combs_array, combs_to_multi_indices


@njit(cache=not CTAPIPE_DISABLE_NUMBA_CACHE)
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


@njit(cache=not CTAPIPE_DISABLE_NUMBA_CACHE)
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


@njit(cache=not CTAPIPE_DISABLE_NUMBA_CACHE)
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
