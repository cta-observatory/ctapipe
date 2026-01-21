#!/usr/bin/env python3

from astropy import units as u
from astropy.table import Table


def make_example_dl2_table():
    return Table(
        dict(
            obs_id=[10, 10, 10, 10],
            event_id=[1, 2, 3, 4],
            true_energy=[100.0, 50.0, 2.0, 30.0] * u.TeV,
            RandomForestRegressor_energy=[100.1, 49.2, 2.6, 40.0] * u.TeV,
            HillasReconstructor_az=[271.0, 271.6, 271.4, 268.1] * u.deg,
            HillasReconstructor_alt=[70.1, 68.2, 69.3, 70.8] * u.deg,
            RandomForestClassifier_prediction=[0.9, 0.5, 0.1, 0.3],
            true_alt=[70.0, 70.0, 70.0, 70.0] * u.deg,
            true_az=[270.0, 270.0, 270.0, 270.0] * u.deg,
            subarray_pointing_lat=[70.0, 70.0, 70.0, 70.0] * u.deg,
            subarray_pointing_lon=[270.0, 270.0, 270.0, 270.0] * u.deg,
        )
    )


def test_event_preprocessing():
    from ctapipe.io.dl2_tables_preprocessing import (
        DL2EventPreprocessorNew,
        DL2FeatureSet,
        get_default_features_to_store,
    )

    table = make_example_dl2_table()

    preprocess = DL2EventPreprocessorNew(feature_set=str(DL2FeatureSet.simulation))
    table_processed = preprocess(table)

    for feature in get_default_features_to_store(DL2FeatureSet.simulation):
        assert feature in table_processed.columns
