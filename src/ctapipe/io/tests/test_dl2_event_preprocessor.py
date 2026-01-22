#!/usr/bin/env python3

import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable

from ctapipe.io.dl2_tables_preprocessing import DL2FeatureSet


def make_example_dl2_table():
    return QTable(
        dict(
            obs_id=[10, 10, 10, 10],
            event_id=[1, 2, 3, 4],
            true_energy=[100.0, 50.0, 2.0, 30.0] * u.TeV,
            RandomForestRegressor_energy=[100.1, 49.2, 2.6, 40.0] * u.TeV,
            RandomForestRegressor_is_valid=[True, True, True, True],
            HillasReconstructor_az=[271.0, 271.6, 271.4, 268.1] * u.deg,
            HillasReconstructor_alt=[70.1, 68.2, 69.3, 70.8] * u.deg,
            HillasReconstructor_is_valid=[True, True, True, False],
            RandomForestClassifier_prediction=[0.9, 0.5, 0.1, 0.3],
            RandomForestClassifier_is_valid=[True, True, True, False],
            true_alt=[70.0, 70.0, 70.0, 70.0] * u.deg,
            true_az=[270.0, 270.0, 270.0, 270.0] * u.deg,
            subarray_pointing_lat=[70.0, 70.0, 70.0, 70.0] * u.deg,
            subarray_pointing_lon=[270.0, 270.0, 270.0, 270.0] * u.deg,
            RandomForestClassifier_telescopes=np.array(
                [
                    [False, True, True, True],
                    [True, True, True, True],
                    [True, True, False, True],
                    [True, True, True, True],
                ]
            ),
        )
    )


@pytest.mark.parametrize("feature_set", list(DL2FeatureSet))
def test_event_preprocessing(feature_set):
    from traitlets.config import Config

    from ctapipe.io.dl2_tables_preprocessing import DL2EventPreprocessor

    # set some custom features for the case where the feature_set==custom.
    # These will be ignored in other feature_sets.
    custom_config = Config()
    custom_config.DL2EventPreprocessor.features = ["obs_id", "event_id"]

    table = make_example_dl2_table()

    # process the table:
    preprocess = DL2EventPreprocessor(config=custom_config, feature_set=feature_set)
    table_processed = preprocess(table)

    for feature in preprocess.features:
        assert feature in table_processed.columns


def test_no_output():
    """Check error is raised if no columns are specified for output."""
    from ctapipe.core import ToolConfigurationError
    from ctapipe.io.dl2_tables_preprocessing import DL2EventPreprocessor

    with pytest.raises(ToolConfigurationError):
        DL2EventPreprocessor(feature_set=DL2FeatureSet.custom)
