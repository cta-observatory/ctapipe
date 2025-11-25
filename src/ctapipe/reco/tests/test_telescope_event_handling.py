import astropy.units as u
import numpy as np
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


def test_calc_combs_min_distances():
    from ctapipe.reco.telescope_event_handling import (
        calc_combs_min_distances_event,
        calc_combs_min_distances_table,
    )

    index_tel_combs = np.array([[0, 1], [0, 2], [1, 2]])
    fov_lon_values = np.array([[1, 2], [3, 4], [5, 6]])
    fov_lat_values = np.array([[7, 8], [9, 10], [11, 12]])
    weights = np.array([1, 2, 3])
    dist_weights = np.array([[1, 1], [1, 1], [1, 0.55]], dtype=np.float64)

    exp_comb_weights = np.array([3, 4, 5])
    exp_fov_lons = np.array([2.66666667, 5, 5.2])
    exp_fov_lats = np.array([8.66666667, 11, 11.2])

    fov_lons_ev, fov_lats_ev, comb_weights_ev = calc_combs_min_distances_event(
        index_tel_combs,
        fov_lon_values,
        fov_lat_values,
        weights,
        dist_weights,
    )

    fov_lons_tab, fov_lats_tab, comb_weights_tab = calc_combs_min_distances_table(
        index_tel_combs,
        fov_lon_values,
        fov_lat_values,
        weights,
        dist_weights,
    )
    assert np.allclose(fov_lons_ev, exp_fov_lons)
    assert np.allclose(fov_lons_tab, exp_fov_lons)
    assert np.allclose(fov_lats_ev, exp_fov_lats)
    assert np.allclose(fov_lats_tab, exp_fov_lats)
    assert np.allclose(comb_weights_ev, exp_comb_weights)
    assert np.allclose(comb_weights_tab, exp_comb_weights)


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
