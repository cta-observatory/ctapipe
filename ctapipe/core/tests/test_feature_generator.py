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

    input_table = Table({"intensity": [1, 10, 100], "length": [2, 4, 8], "width": [1, 2, 4]})
    input_columns = list(input_table.colnames)
    new_columns = [name for name, _ in expressions]

    log_intensity = [0, 1, 2]
    area = [2, 8, 32]
    eccentricity = np.sqrt(0.75)

    new_features = generator(input_table)

    # check we don't modify the input column and only get the new new_features
    # in the output column
    for column in input_columns:
        assert column not in new_features.colnames
        assert column in input_table.colnames

    for column in new_columns:
        assert column not in input_table.colnames
        assert column in new_features.colnames

    assert np.all(new_features["log_intensity"] == log_intensity)
    assert np.all(new_features["area"] == area)
    assert np.all(new_features["eccentricity"] == eccentricity)


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
    """Test chaning the unit of a feature"""
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
