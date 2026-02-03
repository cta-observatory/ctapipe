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
    Get subarray-event-wise information from a table of telescope events.

    Extract the unique subarray events contained in a table of telescope-event
    rows and return their observation IDs, event IDs, multiplicities, and an
    index mapping from each telescope-event row to its corresponding subarray
    event.

    This requires that all telescope events belonging to the same subarray
    event appear in a single contiguous block in ``tel_table``.

    Parameters
    ----------
    tel_table : astropy.table.Table
        Table with telescope events as rows. Must contain the columns
        ``obs_id`` and ``event_id`` and be ordered such that telescope rows
        of the same (obs_id, event_id) are contiguous.

    Returns
    -------
    obs_ids : np.ndarray
        Observation IDs of the subarray events. One entry per subarray event.
    event_ids : np.ndarray
        Event IDs of the subarray events. One entry per subarray event.
    multiplicity : np.ndarray
        Number of telescope-event rows contributing to each subarray event.
    tel_to_array_index : np.ndarray
        Integer array mapping each telescope-event row to the corresponding
        subarray-event index. Has the same length as ``tel_table``.
    """
    obs_idx = tel_table["obs_id"]
    event_idx = tel_table["event_id"]
    return _get_subarray_index(obs_idx, event_idx)


def _grouped_add(tel_data, n_array_events, indices):
    """
    Sum telescope-event values per subarray event.

    Groups telescope-event values by their corresponding subarray-event index
    and computes the group-wise sum using ``np.add.at``.

    Parameters
    ----------
    tel_data : np.ndarray
        Values for each telescope event (one value per telescope-event row).
    n_array_events : int
        Total number of subarray events (size of the grouped output).
    indices : np.ndarray
        Integer array mapping each telescope-event row to its corresponding
        subarray-event index. Must have the same length as ``tel_data``.

    Returns
    -------
    summed : np.ndarray
        Array of shape ``(n_array_events,)`` containing the sum of ``tel_data``
        over all telescope-event rows belonging to each subarray event.
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
    Compute weighted mean and standard deviation per subarray  event.

    Telescope-event values (``tel_values``) are grouped by subarray event using
    ``indices`` (mapping each telescope-event row to an subarray-event index).
    For each subarray event, the function computes the weighted mean and the
    weighted standard deviation over the corresponding telescope-event rows.

    Invalid telescope rows are excluded using ``valid_tel``. Subarray events
    without any valid telescope contribution return ``NaN`` for both mean and
    standard deviation.

    Parameters
    ----------
    tel_values : np.ndarray
        Values for each telescope event (one value per telescope-event row).
    valid_tel : array-like
        Boolean mask selecting valid telescope-event rows in ``tel_values``.
        Must have the same length as ``tel_values``.
    indices : np.ndarray
        Integer array mapping each telescope-event row to the corresponding
        subarray-event index. This is the fourth return value of
        ``get_subarray_index``. Must have the same length as ``tel_values``.
    multiplicity : np.ndarray
        Number of telescope-event rows per subarray event (one entry per
        subarray event), in the same order as the subarray events encoded in
        ``indices``.
    weights : np.ndarray, optional
        Weights used for averaging. Must be broadcastable to ``tel_values``.
        If not provided, equal weights are used.

    Returns
    -------
    mean : np.ndarray
        Weighted mean value for each subarray event. Entries are
        ``NaN`` for events without valid telescope contributions.
    std : np.ndarray
        Weighted standard deviation for each subarray event.
        Entries are ``NaN`` where the mean is undefined.
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
    Generate all index combinations of a fixed size.

    Returns all ``comb_size``-element combinations chosen from the index range
    ``0 .. array_length - 1``. Results are cached to speed up repeated calls
    with the same arguments.

    Parameters
    ----------
    array_length : int
        Length of the index range to draw from.
    comb_size : int
        Size of each combination.

    Returns
    -------
    combs : np.ndarray
        Integer array of shape ``(n_combs, comb_size)`` containing all index
        combinations, where ``n_combs = binom(array_length, comb_size)``.
    """
    return np.array(list(combinations(range(array_length), comb_size)))


@lru_cache(maxsize=4096)
def binary_combinations(k: int) -> np.ndarray:
    """
    Generate all binary (0/1) vectors of length ``k``.

    Returns the full set of binary sign assignments used e.g. to enumerate
    DISP head–tail sign choices. Results are cached to speed up repeated calls.

    Parameters
    ----------
    k : int
        Length of each binary vector.

    Returns
    -------
    combs : np.ndarray
        Integer array of shape ``(2**k, k)`` containing all possible binary
        combinations. Each row is one assignment.
    """
    return np.array(list(product([0, 1], repeat=k)), dtype=int)


def check_ang_diff(min_ang_diff, psi1, psi2):
    """
    Check whether two Hillas main-axis orientations are sufficiently separated.

    Computes an axis-invariant angular difference between two orientation angles,
    treating directions that differ by 180° as parallel. The resulting difference
    is folded into the interval [0°, 90°] and compared against ``min_ang_diff``.

    Parameters
    ----------
    min_ang_diff : float
        Minimum required angular separation in degrees.
    psi1, psi2 : astropy.units.Quantity
        Orientation angles of the Hillas main axes. Scalars or array-like objects
        of identical shape with angular units (e.g. degrees).

    Returns
    -------
    keep : bool or np.ndarray
        Boolean value or boolean array indicating whether the (axis-invariant)
        angular separation is greater than or equal to ``min_ang_diff``.
        ``True`` means the event should be kept; ``False`` means the axes are
        too parallel.
    """
    ang_diff = np.abs(psi1 - psi2) % (180 * u.deg)
    ang_diff = np.minimum(ang_diff, 180 * u.deg - ang_diff)
    return ang_diff >= (min_ang_diff * u.deg)


def calc_combs_min_distances(index_tel_combs, fov_lon_values, fov_lat_values, weights):
    """
    Select the DISP sign assignment that minimizes the weighted SSE per combination.

    For each telescope combination (size ``k``), evaluate all ``2**k`` possible
    binary sign assignments (DISP head–tail ambiguity). For each assignment,
    compute the weighted mean FoV direction and the weighted sum of squared
    distances (SSE) of the telescope directions to this mean. The assignment
    with minimal SSE is chosen.

    The function returns the weighted mean FoV longitude/latitude for the best
    assignment of each telescope combination, plus the summed weight per
    combination.

    Parameters
    ----------
    index_tel_combs : np.ndarray
        Integer array of shape ``(n_combs, k)`` containing index combinations of
        telescope events forming each combination.
    fov_lon_values : np.ndarray
        Array of shape ``(n_tel_events, 2)`` containing the two possible FoV
        longitude values (SIGN = 0/1, i.e. DISP sign choices) for each telescope
        event, in degrees.
    fov_lat_values : np.ndarray
        Array of shape ``(n_tel_events, 2)`` containing the two possible FoV
        latitude values (SIGN = 0/1) for each telescope event, in degrees.
    weights : np.ndarray
        Array of shape ``(n_tel_events,)`` containing the weight per telescope
        event.

    Returns
    -------
    weighted_lons : np.ndarray
        Array of shape ``(n_combs,)`` with the weighted mean FoV longitudes of
        the best sign assignment for each telescope combination.
    weighted_lats : np.ndarray
        Array of shape ``(n_combs,)`` with the weighted mean FoV latitudes of
        the best sign assignment for each telescope combination.
    combined_weights : np.ndarray
        Array of shape ``(n_combs,)`` containing the sum of telescope weights
        contributing to each combination.
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
    subarray events with a given multiplicity.

    The function assumes that telescope events are grouped contiguously
    by subarray event, i.e. all rows belonging to the same subarray event appear
    in a single consecutive block in ``valid_tel_to_array_indices``.
    Under this assumption, the multiplicity of an subarray event corresponds
    to the length of its contiguous block.

    For all subarray events whose block length equals ``multi``, the function
    returns a boolean mask that is ``True`` for all telescope rows belonging
    to those events and ``False`` otherwise.

    This block-based approach avoids repeated membership tests (e.g.
    ``np.isin``) and is faster for large arrays.

    Parameters
    ----------
    multi : int
        Target multiplicity (number of telescope events per subarray event).
    valid_tel_to_array_indices : np.ndarray
        One-dimensional array mapping each valid telescope event to its
        corresponding subarray-event index. Must be ordered such that
        telescope events of the same subarray event are contiguous.

    Returns
    -------
    mask : np.ndarray
        Boolean array of the same length as ``valid_tel_to_array_indices``.
        Entries are ``True`` for telescope-event rows belonging to subarray
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
    Fill stereo FoV estimates for events with multiplicity < ``n_tel_combinations``.

    For subarray events whose multiplicity is smaller than ``n_tel_combinations``
    but at least 2, recompute the stereo direction using *all* available
    telescopes of the event (i.e. combination size equals the event multiplicity).
    The optimal DISP sign assignment is determined via
    :func:`calc_combs_min_distances`.

    Results are written in-place into ``fov_lon_combs_mean`` and
    ``fov_lat_combs_mean`` at the positions of the affected events.

    Parameters
    ----------
    fov_lon_combs_mean : np.ndarray
        Array of shape ``(n_array_events,)`` holding the mean FoV longitudes per
        subarray event. Updated in-place for lower-multiplicity events.
    fov_lat_combs_mean : np.ndarray
        Array of shape ``(n_array_events,)`` holding the mean FoV latitudes per
        subarray event. Updated in-place for lower-multiplicity events.
    n_tel_combinations : int
        Nominal number of telescopes used per stereo combination.
    valid_tel_to_array_indices : np.ndarray
        One-dimensional array mapping each valid telescope-event row to its
        corresponding subarray-event index. Telescope rows must be grouped
        contiguously by subarray event.
    valid_multiplicity : np.ndarray
        Multiplicity per valid subarray event (one entry per subarray event), aligned
        with the indices used in ``valid_tel_to_array_indices``.
    fov_lon_values : np.ndarray
        Array of shape ``(n_valid_tel_events, 2)`` containing the two possible
        FoV longitude values (DISP sign choices) for each valid telescope event.
    fov_lat_values : np.ndarray
        Array of shape ``(n_valid_tel_events, 2)`` containing the two possible
        FoV latitude values (DISP sign choices) for each valid telescope event.
    weights : np.ndarray
        Array of shape ``(n_valid_tel_events,)`` with the weight per valid
        telescope event.

    Returns
    -------
    None
        Operates in-place and does not return a value.
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
    Compute the two possible FoV longitude/latitude coordinates per telescope event.

    For each telescope event, compute two candidate shower directions in the
    telescope FoV corresponding to the two possible DISP sign assignments
    (SIGN = −1 and +1). The candidates are obtained by shifting the Hillas
    centroid along the main shower axis by the absolute DISP distance.

    Parameters
    ----------
    tel_table : astropy.table.Table
        Table containing Hillas parameters and DISP reconstruction results.
        Must include:
        - ``hillas_fov_lon`` : Hillas centroid longitude (Quantity)
        - ``hillas_fov_lat`` : Hillas centroid latitude (Quantity)
        - ``hillas_psi`` : Hillas orientation angle (Quantity)
        - ``<prefix>_parameter`` : DISP distance from the centroid (float)
    prefix : str, optional
        Prefix used to access the DISP parameter column
        (default: "DispReconstructor_tel").

    Returns
    -------
    fov_lon : np.ndarray
        Array of shape ``(n_tel_events, 2)`` containing the candidate FoV
        longitudes (in degrees) for each telescope event (DISP signs −1 and +1).
    fov_lat : np.ndarray
        Array of shape ``(n_tel_events, 2)`` containing the candidate FoV
        latitudes (in degrees) for each telescope event (DISP signs −1 and +1).
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
    Precompute all possible ``k``-combinations for multiplicities up to
    ``max_multiplicity``.

    For each multiplicity ``m`` in the range ``k .. max_multiplicity``, this
    function generates all index combinations of size ``k`` chosen from
    ``0 .. m-1`` (using ``get_combinations(m, k)``) and concatenates them into
    a single array.

    In addition, it returns an array mapping each combination row to the
    multiplicity ``m`` it was generated from. This mapping can later be used
    to quickly select the correct subset of combinations for a given event
    multiplicity without recomputing combinations.

    Parameters
    ----------
    max_multiplicity : int
        Maximum multiplicity to consider (inclusive).
    k : int
        Size of the combinations (must satisfy ``k >= 2`` and
        ``k <= max_multiplicity``).

    Returns
    -------
    combs_array : np.ndarray
        Array of shape ``(n_total_combinations, k)`` containing all possible
        ``k``-combinations for multiplicities ranging from ``k`` up to
        ``max_multiplicity``. Combinations are ordered by increasing
        multiplicity.
    combs_to_multi_indices : np.ndarray
        One-dimensional integer array mapping each row in ``combs_array`` to
        the multiplicity it was generated from. Has the same length as
        ``combs_array``.
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
    Compute the number of ``k``-combinations for each multiplicity.

    Parameters
    ----------
    multiplicity : np.ndarray
        One-dimensional array of multiplicities (number of telescope-event rows)
        for each subarray event.
    k : int
        Size of the combinations.

    Returns
    -------
    n_combs : np.ndarray
        One-dimensional integer array containing the number of possible
        ``k``-combinations for each entry in ``multiplicity``.
    """
    n_combs = np.empty(len(multiplicity), dtype=np.int64)
    for i in range(len(multiplicity)):
        n_combs[i] = _binomial(multiplicity[i], k)

    return n_combs


@njit(cache=not CTAPIPE_DISABLE_NUMBA_CACHE)
def get_index_combs(multiplicities, combs_array, combs_to_multi_indices, k):
    """
    Build telescope-event index combinations for all subarray events.

    For each subarray event with multiplicity ``m = multiplicities[i]``, this
    function selects the precomputed ``k``-combinations corresponding to ``m``
    from ``combs_array`` (using ``combs_to_multi_indices``) and offsets them
    into the global telescope-event indexing scheme.

    The global telescope-event ordering is assumed to be the concatenation of
    telescope-event blocks per subarray event. Under this assumption, the start
    index of event ``i`` equals the cumulative sum of previous multiplicities,
    and combinations for event ``i`` can be obtained by adding that offset.

    Parameters
    ----------
    multiplicities : np.ndarray
        One-dimensional array of multiplicities (number of telescope-event rows)
        for each subarray event.
    combs_array : np.ndarray
        Precomputed combination indices for multiplicities in the range
        ``k .. max_multiplicity``. Usually produced by ``create_combs_array``.
        Shape: ``(n_total_combinations, k)``.
    combs_to_multi_indices : np.ndarray
        One-dimensional array mapping each row in ``combs_array`` to the
        multiplicity it was generated from. Same length as ``combs_array``.
    k : int
        Size of the combinations.

    Returns
    -------
    index_tel_combs : np.ndarray
        Array of shape ``(n_total_combinations_over_events, k)`` containing the
        telescope-event indices for all ``k``-combinations across all subarray
        events. Indices refer to the flattened telescope-event ordering.
    n_combs : np.ndarray
        One-dimensional array giving the number of ``k``-combinations for each
        subarray event, in the same order as ``multiplicities``.
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
