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
            "prediction": [1, 2, 0, 5, 10, 1],
            "dummy_weight": [1, 1, 1, 1, 1, 1],
            "weight": [0.5, 0.5, 0, 4, 1, 1],
        }
    )


def test_mean_prediction(mono_table):
    from ctapipe.ml.stereo_combination import StereoMeanCombiner

    combine_no_weights = StereoMeanCombiner(mono_prediction_column="prediction")
    stereo_no_weights = combine_no_weights(mono_table)
    assert_array_equal(stereo_no_weights, np.array([1, 7.5, 1]))

    combine_dummy_weights = StereoMeanCombiner(
        mono_prediction_column="prediction", weight_column="dummy_weight"
    )
    stereo_dummy_weights = combine_no_weights(mono_table)
    assert_array_equal(stereo_no_weights, stereo_dummy_weights)

    combine_weights = StereoMeanCombiner(
        mono_prediction_column="prediction", weight_column="weight"
    )
    stereo_weights = combine_weights(mono_table)
    assert_array_equal(stereo_weights, np.array([1.5, 6, 1]))


@pytest.mark.skip("Not implemented as of now")
def test_mean_with_quality_query():
    return False
