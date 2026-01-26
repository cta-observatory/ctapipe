#!/usr/bin/env python3

import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable

from ctapipe.io import PreprocessorFeatureSet


@pytest.fixture(scope="function")
def minimal_dl2_table():
    """A dunmmy DL2 table for testing DL2EventPreprocessor"""
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


@pytest.mark.parametrize("feature_set", list(PreprocessorFeatureSet))
def test_event_preprocessing(feature_set, minimal_dl2_table):
    from traitlets.config import Config

    from ctapipe.io import EventPreprocessor

    # set some custom features for the case where the feature_set==custom.
    # These will be ignored in other feature_sets.
    custom_config = Config()
    custom_config.EventPreprocessor.features = ["obs_id", "event_id"]
    table = minimal_dl2_table

    # process the table:
    preprocess = EventPreprocessor(config=custom_config, feature_set=feature_set)
    table_processed = preprocess(table)

    for feature in preprocess.features:
        assert feature in table_processed.columns

    # check that the qualityquery worked
    assert len(table_processed) <= len(table)


def test_no_output():
    """Check error is raised if no columns are specified for output."""
    from ctapipe.core import ToolConfigurationError
    from ctapipe.io import EventPreprocessor, PreprocessorFeatureSet

    with pytest.raises(ToolConfigurationError):
        EventPreprocessor(feature_set=PreprocessorFeatureSet.custom)


def test_nondefault_reconstructors(minimal_dl2_table):
    """Check that using a different constructor than default still works"""

    from ctapipe.io import EventPreprocessor

    # define some new reconstructors, and add those columns to the test table:
    geom = "ExampleGeometryReconstructor"
    energy = "ExampleEnergyRegressor"
    gammaness = "ExampleGammnessClassifier"
    table = minimal_dl2_table

    table[f"{geom}_alt"] = ([71.1, 62.2, 61.3, 75.8] * u.deg,)
    table[f"{geom}_az"] = [231.0, 231.6, 231.4, 238.1] * u.deg
    table[f"{geom}_is_valid"] = [True, False, True, True]

    table[f"{energy}_energy"] = [20.0, 1.0, 0.5, 0.1] * u.TeV
    table[f"{energy}_is_valid"] = [True, False, True, True]

    table[f"{gammaness}_prediction"] = [0.1, 0.8, 0.9, 0.7]
    table[f"{gammaness}_is_valid"] = [True, False, True, True]
    table[f"{gammaness}_telescopes"] = table["RandomForestClassifier_telescopes"]

    preprocess = EventPreprocessor(
        feature_set="dl2_simulation",
        geometry_reconstructor=geom,
        energy_reconstructor=energy,
        gammaness_reconstructor=gammaness,
    )

    table_processed = preprocess(table)

    # check that the processing worked. In this case, we check that the
    # requested columns are renamed correctly and that the filtered values match
    # the original values.

    mask = table["event_id"] == table_processed["event_id"]
    masked = table[mask]  # so that we just compare values after filtering

    assert np.allclose(table_processed["reco_energy"], masked[f"{energy}_energy"])
    assert np.allclose(table_processed["reco_az"], masked[f"{geom}_az"])
    assert np.allclose(table_processed["gh_score"], masked[f"{gammaness}_prediction"])
