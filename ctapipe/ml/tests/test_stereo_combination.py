import pytest
from astropy.table import Table
import astropy.units as u
import numpy as np
from numpy.testing import assert_array_equal


@pytest.fixture(scope="module")
def mono_table():
    """
    Dummy table of telescope events with a
    prediction and weights.
    """
    return Table(
        {
            "obs_id": [1, 1, 1, 1, 1, 2],
            "event_id": [1, 1, 1, 2, 2, 1],
            "tel_id": [1, 2, 3, 5, 7, 1],
            "prediction": u.Quantity([1, 5, 0, 5, 10, 1], u.TeV),
            "dummy_weight": [1, 1, 1, 1, 1, 1],
            "weight": [0.5, 0.5, 0, 4, 1, 1],
        }
    )


def test_mean_prediction(mono_table):
    from ctapipe.ml.stereo_combination import StereoMeanCombiner

    combine_no_weights = StereoMeanCombiner(mono_prediction_column="prediction")
    stereo_no_weights = combine_no_weights(mono_table)
    assert stereo_no_weights.colnames == ["obs_id", "event_id", "mean_prediction"]
    assert_array_equal(stereo_no_weights["obs_id"], np.array([1, 1, 2]))
    assert_array_equal(stereo_no_weights["event_id"], np.array([1, 2, 1]))
    print(mono_table)
    print(stereo_no_weights)
    assert_array_equal(
        stereo_no_weights["mean_prediction"], u.Quantity(np.array([2, 7.5, 1]), u.TeV)
    )

    combine_dummy_weights = StereoMeanCombiner(
        mono_prediction_column="prediction", weight_column="dummy_weight"
    )
    stereo_dummy_weights = combine_dummy_weights(mono_table)
    assert_array_equal(stereo_no_weights, stereo_dummy_weights)

    combine_weights = StereoMeanCombiner(
        mono_prediction_column="prediction", weight_column="weight"
    )
    stereo_weights = combine_weights(mono_table)
    assert_array_equal(
        stereo_weights["mean_prediction"], u.Quantity(np.array([3, 6, 1]), u.TeV)
    )
