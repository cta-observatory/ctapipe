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


def test_mean_std_ufunc(dl1_parameters_file):
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
