import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table

from ctapipe.io import TableLoader, read_table
from ctapipe.io.tests.test_table_loader import check_equal_array_event_order


def test_get_subarray_index(dl1_parameters_file):
    from ctapipe.reco.telescope_event_handling import get_subarray_index

    opts = dict(simulated=False, true_parameters=False, dl2=False, pointing=False)
    with TableLoader(dl1_parameters_file, **opts) as loader:
        tel_events = loader.read_telescope_events()

    subarray_obs_ids, subarray_event_ids, _, _ = get_subarray_index(tel_events)
    trigger = read_table(dl1_parameters_file, "/dl1/event/subarray/trigger")

    assert len(subarray_obs_ids) == len(subarray_event_ids)
    assert len(subarray_obs_ids) == len(trigger)
    check_equal_array_event_order(
        Table({"obs_id": subarray_obs_ids, "event_id": subarray_event_ids}), trigger
    )


def test_weighted_mean_std_ufunc(dl1_parameters_file):
    from ctapipe.reco.telescope_event_handling import (
        get_subarray_index,
        weighted_mean_std_ufunc,
    )

    opts = dict(simulated=False, true_parameters=False, dl2=False, pointing=False)
    with TableLoader(dl1_parameters_file, **opts) as loader:
        tel_events = loader.read_telescope_events()

    valid = np.isfinite(tel_events["hillas_length"])

    _, _, multiplicity, tel_to_subarray_idx = get_subarray_index(tel_events)

    # test only default uniform weights,
    # other weights are tested in test_stereo_combination
    mean, std = weighted_mean_std_ufunc(
        tel_events["hillas_length"], valid, tel_to_subarray_idx, multiplicity
    )

    # check if result is identical with np.mean/np.std
    grouped = tel_events.group_by(["obs_id", "event_id"])
    true_mean = grouped["hillas_length"].groups.aggregate(np.nanmean)
    true_std = grouped["hillas_length"].groups.aggregate(np.nanstd)

    assert np.allclose(mean, true_mean, equal_nan=True)
    assert np.allclose(std, true_std, equal_nan=True)


def test_get_combinations():
    from ctapipe.reco.telescope_event_handling import get_combinations

    tel_ids = [1, 2, 3]
    comb_size = 2
    get_combinations.cache_clear()
    index_combinations = get_combinations(len(tel_ids), comb_size)

    expected_combinations = np.array([[0, 1], [0, 2], [1, 2]])

    assert np.allclose(index_combinations, expected_combinations)


def test_calc_combs_min_distances_multiple_events():
    from ctapipe.reco.telescope_event_handling import calc_combs_min_distances

    index_tel_combs = np.array(
        [
            [0, 1, 2],
            [3, 4, 5],
            [3, 4, 6],
            [3, 5, 6],
            [4, 5, 6],
            [7, 8, 9],
            [7, 8, 10],
            [7, 8, 11],
            [7, 9, 10],
            [7, 9, 11],
            [7, 10, 11],
            [8, 9, 10],
            [8, 9, 11],
            [8, 10, 11],
            [9, 10, 11],
        ]
    )
    tel_ids = np.arange(12)
    fov_lon_values = np.column_stack((tel_ids, tel_ids)).astype(float)
    fov_lat_values = np.column_stack((tel_ids + 100, tel_ids + 100)).astype(float)
    weights = tel_ids + 1

    fov_lons, fov_lats, comb_weights = calc_combs_min_distances(
        index_tel_combs, fov_lon_values, fov_lat_values, weights
    )

    expected_lons = np.array(
        [
            np.average(tel_ids[combo], weights=weights[combo])
            for combo in index_tel_combs
        ]
    )
    expected_lats = np.array(
        [
            np.average(tel_ids[combo] + 100, weights=weights[combo])
            for combo in index_tel_combs
        ]
    )
    expected_weights = np.array([weights[combo].sum() for combo in index_tel_combs])

    assert np.allclose(fov_lons, expected_lons)
    assert np.allclose(fov_lats, expected_lats)
    assert np.allclose(comb_weights, expected_weights)


def test_calc_fov_lon_lat():
    from ctapipe.reco.telescope_event_handling import calc_fov_lon_lat

    prefix = "disp"
    tel_table = Table(
        {
            "hillas_fov_lon": [1, 2, 3] * u.deg,
            "hillas_fov_lat": [4, 5, 6] * u.deg,
            "hillas_psi": [0, 45, 90] * u.deg,
            f"{prefix}_parameter": [1, 2.5, 3] * u.deg,
        }
    )

    lon, lat = calc_fov_lon_lat(tel_table, prefix)

    exp_lon = np.array([[0.0, 2.0], [0.23223305, 3.76776695], [3.0, 3.0]])
    exp_lat = np.array([[4.0, 4.0], [3.23223305, 6.76776695], [3.0, 9.0]])

    assert np.allclose(lon, exp_lon)
    assert np.allclose(lat, exp_lat)


def test_create_combs_array():
    from ctapipe.reco.telescope_event_handling import create_combs_array

    max_multi = 3
    k = 2

    combs_array, combs_to_multi_indices = create_combs_array(max_multi, k)

    exp_combs_array = np.array([[0, 1], [0, 1], [0, 2], [1, 2]])
    exp_combs_to_multi_indices = np.array([2, 3, 3, 3])

    assert np.allclose(combs_array, exp_combs_array)
    assert np.allclose(combs_to_multi_indices, exp_combs_to_multi_indices)


def test_get_index_combs():
    from ctapipe.reco.telescope_event_handling import get_index_combs

    multiplicities = np.array([2, 1, 3, 2])
    combs_array = np.array([[0, 1], [0, 1], [0, 2], [1, 2]])
    combs_to_multi_indices = np.array([2, 3, 3, 3])
    k = 2

    index_tel_combs, num_combs = get_index_combs(
        multiplicities, combs_array, combs_to_multi_indices, k
    )

    exp_index_tel_combs = np.array([[0, 1], [3, 4], [3, 5], [4, 5], [6, 7]])
    exp_num_combs = np.array([1, 0, 3, 1])

    assert np.allclose(index_tel_combs, exp_index_tel_combs)
    assert np.allclose(num_combs, exp_num_combs)


@pytest.mark.parametrize(
    ("k", "expected"),
    [
        (2, np.array([[0, 0], [0, 1], [1, 0], [1, 1]])),
        (0, np.empty((1, 0), dtype=int)),
    ],
)
def test_binary_combinations(k, expected):
    from ctapipe.reco.telescope_event_handling import binary_combinations

    combinations = binary_combinations(k)

    assert combinations.shape == expected.shape
    assert np.array_equal(combinations, expected)


@pytest.mark.parametrize(
    ("min_ang_diff", "psi1", "psi2", "expected"),
    [
        (20.0, 10.0 * u.deg, 50.0 * u.deg, True),
        (10.0, 0.0 * u.deg, 180.0 * u.deg, False),
    ],
)
def test_check_ang_diff(min_ang_diff, psi1, psi2, expected):
    from ctapipe.reco.telescope_event_handling import check_ang_diff

    result = check_ang_diff(min_ang_diff, psi1, psi2)

    assert bool(result) is expected


def test_check_ang_diff_array():
    from ctapipe.reco.telescope_event_handling import check_ang_diff

    min_ang_diff = 20.0
    psi1 = np.array([0.0, 10.0, 0.0]) * u.deg
    psi2 = np.array([30.0, 25.0, 180.0]) * u.deg

    result = check_ang_diff(min_ang_diff, psi1, psi2)
    expected = np.array([True, False, False])

    assert np.all(result == expected)


def test_valid_tels_of_multi():
    from ctapipe.reco.telescope_event_handling import valid_tels_of_multi

    valid_tel_to_array_indices = np.array([0, 0, 1, 1, 1, 2])

    mask = valid_tels_of_multi(2, valid_tel_to_array_indices)
    expected = np.array([True, True, False, False, False, False])

    assert np.array_equal(mask, expected)


def test_valid_tels_of_multi_noop():
    from ctapipe.reco.telescope_event_handling import valid_tels_of_multi

    valid_tel_to_array_indices = np.array([0, 0, 1, 1, 1, 2])

    mask = valid_tels_of_multi(4, valid_tel_to_array_indices)

    assert not np.any(mask)


def test_fill_lower_multiplicities():
    from ctapipe.reco.telescope_event_handling import fill_lower_multiplicities

    fov_lon_combs_mean = np.array([10.0, np.nan, 30.0])
    fov_lat_combs_mean = np.array([0.0, np.nan, 1.0])
    n_tel_combinations = 3
    valid_tel_to_array_indices = np.array([0, 0, 0, 1, 1, 2])
    valid_multiplicity = np.array([3, 2, 1])
    fov_lon_values = np.array(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 11.0],
            [2.0, 12.0],
            [5.0, 15.0],
        ]
    )
    fov_lat_values = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )
    weights = np.ones(len(valid_tel_to_array_indices))

    fill_lower_multiplicities(
        fov_lon_combs_mean,
        fov_lat_combs_mean,
        n_tel_combinations,
        valid_tel_to_array_indices,
        valid_multiplicity,
        fov_lon_values,
        fov_lat_values,
        weights,
    )

    assert np.allclose(fov_lon_combs_mean, np.array([10.0, 1.5, 30.0]), equal_nan=True)
    assert np.allclose(fov_lat_combs_mean, np.array([0.0, 0.0, 1.0]), equal_nan=True)


def test_fill_lower_multiplicities_noop():
    from ctapipe.reco.telescope_event_handling import fill_lower_multiplicities

    fov_lon_combs_mean = np.array([10.0, 20.0])
    fov_lat_combs_mean = np.array([0.0, 1.0])
    n_tel_combinations = 2
    valid_tel_to_array_indices = np.array([0, 0, 1, 1])
    valid_multiplicity = np.array([2, 2])
    fov_lon_values = np.array([[0.0, 1.0], [0.0, 1.0], [2.0, 3.0], [2.0, 3.0]])
    fov_lat_values = np.zeros_like(fov_lon_values)
    weights = np.ones(len(valid_tel_to_array_indices))

    fill_lower_multiplicities(
        fov_lon_combs_mean,
        fov_lat_combs_mean,
        n_tel_combinations,
        valid_tel_to_array_indices,
        valid_multiplicity,
        fov_lon_values,
        fov_lat_values,
        weights,
    )

    assert np.allclose(fov_lon_combs_mean, np.array([10.0, 20.0]))
    assert np.allclose(fov_lat_combs_mean, np.array([0.0, 1.0]))
