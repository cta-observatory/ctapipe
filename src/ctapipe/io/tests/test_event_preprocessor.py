#!/usr/bin/env python3

import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable
from astropy.time import Time

from ctapipe.io.event_preprocessor import EventPreprocessorMode, FeatureSetRegistry


@pytest.fixture(scope="function")
def minimal_dl2_table():
    """A dunmmy DL2 table for testing DL2EventPreprocessor"""
    return QTable(
        dict(
            obs_id=[10, 10, 10, 10],
            event_id=[1, 2, 3, 4],
            true_energy=[100.0, 50.0, 2.0, 30.0] * u.TeV,
            time=Time([0, 0, 0, 0], format="unix", scale="utc"),
            RandomForestRegressor_energy=[100.1, 49.2, 2.6, 40.0] * u.TeV,
            RandomForestRegressor_is_valid=[True, True, True, True],
            HillasReconstructor_az=[271.0, 271.6, 271.4, 268.1] * u.deg,
            HillasReconstructor_alt=[70.1, 68.2, 69.3, 70.8] * u.deg,
            HillasReconstructor_h_max=[2000.0, 2500.0, 1980.0, 1000.0] * u.m,
            HillasReconstructor_is_valid=[True, True, True, False],
            HillasReconstructor_telescopes=np.ones(shape=(4, 13), dtype=bool),
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


@pytest.mark.parametrize("mode", list(EventPreprocessorMode))
@pytest.mark.parametrize("feature_set", FeatureSetRegistry.list_available())
def test_event_preprocessing(
    feature_set, mode, minimal_dl2_table, subarray_prod5_paranal
):
    from traitlets.config import Config

    from ctapipe.io import EventPreprocessor

    # set some custom features for the case where the feature_set==custom.
    # These will be ignored in other feature_sets.
    custom_config = Config()
    custom_config.EventPreprocessor.features = ["obs_id", "event_id"]
    table = minimal_dl2_table

    # process the table:
    preprocess = EventPreprocessor(
        config=custom_config,
        feature_set=feature_set,
        subarray=subarray_prod5_paranal,
        mode=mode,
    )
    table_processed = preprocess(
        table,
        gammaness_cut_function=lambda g, e: True,
        multiplicity_cut_function=lambda m, e: m >= 4,
    )

    for feature in preprocess.features:
        assert feature in table_processed.columns

    if mode == EventPreprocessorMode.DROP:
        # check that the qualityquery worked
        assert len(table_processed) <= len(table)

    if mode == EventPreprocessorMode.MARK:
        # check that we have all the original events in KEEP mode
        assert len(table_processed) == len(table)

        # check that new columns were added:
        assert set(preprocess.quality_query.criteria_names) <= set(
            table_processed.colnames
        )


def test_dl3_fov_coordinates(minimal_dl2_table, subarray_prod5_paranal):
    """Check that DETX and DETY follow the GADF sign convention."""
    from ctapipe.io import EventPreprocessor

    preprocess = EventPreprocessor(
        feature_set="dl2_to_dl3",
        subarray=subarray_prod5_paranal,
        mode=EventPreprocessorMode.MARK,
    )
    table = preprocess(
        minimal_dl2_table,
        gammaness_cut_function=lambda g, e: True,
        multiplicity_cut_function=lambda m, e: True,
    )

    # the pointing is at az=270 deg, alt=70 deg: DETX increases with decreasing
    # azimuth, and DETY with increasing altitude
    assert np.all(np.sign(table["DETX"]) == [-1, -1, -1, 1])
    assert np.all(np.sign(table["DETY"]) == [1, -1, -1, 1])


def test_dl3_multiplicity(minimal_dl2_table, subarray_prod5_paranal):
    """Check that MULTIP has the dtype and values of the telescope multiplicity."""
    from ctapipe.io import EventPreprocessor

    preprocess = EventPreprocessor(
        feature_set="dl2_to_dl3",
        subarray=subarray_prod5_paranal,
        mode=EventPreprocessorMode.MARK,
    )
    table = preprocess(
        minimal_dl2_table,
        gammaness_cut_function=lambda g, e: True,
        multiplicity_cut_function=lambda m, e: True,
    )

    expected = subarray_prod5_paranal.multiplicity(
        minimal_dl2_table["HillasReconstructor_telescopes"]
    )
    assert table["MULTIP"].dtype == np.int16
    assert np.all(table["MULTIP"] == expected)


def test_dl3_matches_datamodel(minimal_dl2_table, subarray_prod5_paranal):
    """Check that the output conforms to the GADF events table of the datamodel."""
    from ctao_datamodel import model_validate_astropy_table
    from ctao_datamodel.models.dl3.ctao_gadf import GADFEventsTable

    from ctapipe.io import EventPreprocessor

    cut_inputs = {}

    def gammaness_cut(gammaness, energy):
        cut_inputs["gammaness"] = gammaness
        cut_inputs["energy"] = energy
        return True

    preprocess = EventPreprocessor(
        feature_set="dl2_to_dl3",
        subarray=subarray_prod5_paranal,
        mode=EventPreprocessorMode.MARK,
    )
    table = preprocess(
        minimal_dl2_table,
        gammaness_cut_function=gammaness_cut,
        multiplicity_cut_function=lambda m, e: True,
    )

    # TIME stays a Time column until the file is written
    assert isinstance(table["TIME"], Time)
    table["TIME"] = table["TIME"].unix * u.s
    model_validate_astropy_table(table, GADFEventsTable)

    # not (yet) covered by the datamodel
    assert table["DETY"].dtype == np.float32
    assert table["HMAX"].dtype == np.float32

    # the cuts see the values before they are reduced to float32
    assert cut_inputs["gammaness"].dtype == np.float64
    assert cut_inputs["energy"].dtype == np.float64


def test_no_output():
    """Check error is raised if no columns are specified for output."""
    from ctapipe.core import ToolConfigurationError
    from ctapipe.io import EventPreprocessor

    with pytest.raises(ToolConfigurationError):
        EventPreprocessor(feature_set="custom")


def test_dl3_requires_subarray():
    """Check error is raised at construction if the subarray is missing."""
    from ctapipe.core import ToolConfigurationError
    from ctapipe.io import EventPreprocessor

    with pytest.raises(ToolConfigurationError, match="requires a subarray"):
        EventPreprocessor(feature_set="dl2_to_dl3")


@pytest.mark.parametrize(
    "given,missing",
    [
        ({}, "gammaness_cut_function.*multiplicity_cut_function"),
        ({"gammaness_cut_function": lambda g, e: True}, "multiplicity_cut_function"),
        ({"multiplicity_cut_function": lambda m, e: True}, "gammaness_cut_function"),
    ],
)
def test_dl3_requires_cut_functions(
    given, missing, minimal_dl2_table, subarray_prod5_paranal
):
    """Check error is raised before generating features if a cut function is missing."""
    from ctapipe.io import EventPreprocessor

    preprocess = EventPreprocessor(
        feature_set="dl2_to_dl3", subarray=subarray_prod5_paranal
    )

    def fail(*args, **kwargs):
        raise AssertionError("features should not be generated")

    preprocess.feature_generator = fail

    with pytest.raises(TypeError, match=missing):
        preprocess(minimal_dl2_table, **given)


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
        feature_set="dl2_irf",
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
