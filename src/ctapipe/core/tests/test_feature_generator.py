""" Tests of Selectors """
import numpy as np
import pytest
from astropy.table import Table

from ctapipe.core.expression_engine import ExpressionError
from ctapipe.core.feature_generator import FeatureGenerator, FeatureGeneratorException


def test_feature_generator():
    """Test if generating features works."""
    expressions = [
        ("log_intensity", "log10(intensity)"),
        ("area", "length * width"),
        ("eccentricity", "sqrt(1 - width ** 2 / length ** 2)"),
    ]

    generator = FeatureGenerator(features=expressions)

    table = Table({"intensity": [1, 10, 100], "length": [2, 4, 8], "width": [1, 2, 4]})
    log_intensity = [0, 1, 2]
    area = [2, 8, 32]
    eccentricity = np.sqrt(0.75)

    table = generator(table)

    assert "log_intensity" in table.colnames
    assert "area" in table.colnames
    assert "eccentricity" in table.colnames

    assert np.all(table["log_intensity"] == log_intensity)
    assert np.all(table["area"] == area)
    assert np.all(table["eccentricity"] == eccentricity)


def test_existing_feature():
    """If the feature already exists, fail"""
    expressions = [("foo", "bar")]
    generator = FeatureGenerator(features=expressions)
    table = Table({"foo": [1], "bar": [1]})

    with pytest.raises(FeatureGeneratorException):
        generator(table)


def test_missing_colname():
    """If the column to create a feature misses, fail"""
    expressions = [("foo", "bar")]
    generator = FeatureGenerator(features=expressions)
    table = Table({"baz": [1]})

    with pytest.raises(ExpressionError):
        generator(table)


def test_to_unit():
    """Test changing the unit of a feature"""
    from astropy import units as u

    expressions = [
        ("length_meter", "length.to(u.m)"),
        ("log_length_meter", "log10(length.quantity.to_value(u.m))"),
    ]
    generator = FeatureGenerator(features=expressions)
    table = Table({"length": [1 * u.km]})

    table = generator(table)
    assert table["length_meter"] == 1000
    assert table["log_length_meter"] == 3


def test_multiplicity(subarray_prod5_paranal):
    """Test if generating features works."""

    expressions = [
        ("n_telescopes", "subarray.multiplicity(tels_with_trigger)"),
        ("n_lsts", "subarray.multiplicity(tels_with_trigger, 'LST_LST_LSTCam')"),
        ("n_msts", "subarray.multiplicity(tels_with_trigger, 'MST_MST_FlashCam')"),
        ("n_ssts", "subarray.multiplicity(tels_with_trigger, 'SST_ASTRI_CHEC')"),
    ]

    subarray = subarray_prod5_paranal.select_subarray([1, 2, 20, 21, 80, 81])

    masks = np.array(
        [
            [True, False, True, True, False, False],
            [True, True, False, True, False, True],
        ]
    )

    table = Table(
        {
            "tels_with_trigger": masks,
        }
    )

    generator = FeatureGenerator(features=expressions)
    table = generator(table, subarray=subarray)

    np.testing.assert_equal(table["n_telescopes"], [3, 4])
    np.testing.assert_equal(table["n_lsts"], [1, 2])
    np.testing.assert_equal(table["n_msts"], [2, 1])
    np.testing.assert_equal(table["n_ssts"], [0, 1])
