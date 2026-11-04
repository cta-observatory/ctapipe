"""Tests of Selectors"""

import numpy as np
import pytest
from astropy.table import QTable, Table
from astropy.time import Time

from ctapipe.core.expression_engine import ExpressionError
from ctapipe.core.qualityquery import QualityCriteriaError, QualityQuery
from ctapipe.core.traits import List


def test_selector():
    """test the functionality of an example Selector subclass"""

    class ExampleQualityQuery(QualityQuery):
        """Available variables: x"""

        quality_criteria = List(
            default_value=[
                ("high_enough", "x > 3"),
                ("a_value_not_too_high", "x < 100"),
                ("smallish", "x < sqrt(100)"),
            ],
        ).tag(config=True)

    query = ExampleQualityQuery()

    criteria1 = query(x=0)  # pass smallish
    assert len(criteria1) == 3
    assert (criteria1 == [False, True, True]).all()

    criteria2 = query(x=20)  # pass high_enough + not_too_high
    assert (criteria2 == [True, True, False]).all()

    criteria3 = query(x=200)  # pass high_enough, fail not_too_high
    assert (criteria3 == [True, False, False]).all()

    criteria4 = query(x=8)  # pass all
    assert np.all(criteria4)

    tab = query.to_table()
    html = query._repr_html_()
    assert isinstance(html, str)

    assert tab["criteria"][0] == "TOTAL"
    assert tab["criteria"][1] == "high_enough"
    assert tab["criteria"][2] == "a_value_not_too_high"
    assert tab["criteria"][3] == "smallish"

    assert tab["counts"][0] == 4
    assert tab["counts"][1] == 3
    assert tab["counts"][2] == 3
    assert tab["counts"][3] == 2

    # 0 0 0
    # 1 1 0
    # 1 0 0
    # 1 1 1
    assert tab["cumulative_counts"][0] == 4
    assert tab["cumulative_counts"][1] == 3
    assert tab["cumulative_counts"][2] == 2
    assert tab["cumulative_counts"][3] == 1


def test_invalid_input():
    class ExampleQualityQuery(QualityQuery):
        """Available variables: x"""

        quality_criteria = List(
            default_value=[
                ("high_enough", "x > 3"),
                ("a_value_not_too_high", "x < 100"),
                ("smallish", "x < np.sqrt(100)"),
            ],
        ).tag(config=True)

    query = ExampleQualityQuery()
    with pytest.raises(ExpressionError):
        query(y=5)


def test_bad_selector():
    """ensure failure if a selector function is not a function or can't be evaluated"""

    query = QualityQuery(
        quality_criteria=[
            ("high_enough", "x > 3"),
            ("not_good", "foo"),
            ("smallish", "x < 10"),
        ]
    )
    with pytest.raises(ExpressionError):
        query(x=5)

    # ensure we can't run arbitrary code.
    # try to construct something that is not in the
    # ALLOWED_GLOBALS list, but which is imported in selector.py
    # and see if it works in a function
    with pytest.raises(ExpressionError):
        query = QualityQuery(quality_criteria=[("dangerous", "Component()")])
        query(x=10)

    # test we only support expressions, not statements
    with pytest.raises(ExpressionError):
        query = QualityQuery(
            quality_criteria=[("dangerous", "import numpy; np.array([])")]
        )


def test_table_mask():
    """Test getting a mask for a whole table"""
    query = QualityQuery(
        quality_criteria=[
            ("foo", "(x**2 + y**2) < 1.0"),
            ("bar", "x < 0.5"),
        ],
    )

    table = Table({"x": [1.0, 0.2, -0.5, 0.6, 0.7], "y": [0.0, 0.5, 1.0, 0.2, 0.1]})

    mask = query.get_table_mask(table)
    assert len(mask) == len(table)
    assert mask.dtype == np.bool_
    np.testing.assert_equal(mask, [False, True, False, False, False])
    stats = query.to_table()
    np.testing.assert_equal(stats["counts"], [5, 3, 2])
    np.testing.assert_equal(stats["cumulative_counts"], [5, 3, 1])


def test_printing():
    """Just check the query can be stringified correctly"""
    query = QualityQuery(
        quality_criteria=[("check", "x>3")],
    )

    assert isinstance(str(query), str)


def test_setup():
    """Test that tuples can only have length 2"""
    from traitlets import TraitError

    # 2-tuple works
    QualityQuery(quality_criteria=[("1", "2")])

    # 3-tuple fails
    with pytest.raises(TraitError):
        QualityQuery(quality_criteria=[("1", "2", "3")])


def test_add_columns():
    """Test that we can add columns."""

    query = QualityQuery(
        quality_criteria=[
            ("low", "x>0"),
            ("high", "x < 10"),
        ],
    )

    table = Table({"x": [-10, 1, 2, 3, 5, 11, 5]})

    table_with_cols = query.add_table_mask_columns(table)

    assert "low" in table_with_cols.colnames
    assert "high" in table_with_cols.colnames
    assert table_with_cols["low"].sum() == 6


@pytest.mark.parametrize(
    "criteria",
    [
        pytest.param([("cut", "True")], id="python_scalar"),
        pytest.param([("cut", "np.True_")], id="numpy_scalar"),
        pytest.param([("cut", "np.asarray(x) > 0")], id="ndarray"),
        pytest.param([("cut", "x > 0")], id="column"),
    ],
)
def test_add_columns_matches_table_mask(criteria):
    """Test that adding columns accepts the same results as get_table_mask."""

    table = Table({"x": [-10, 1, 11]})

    query = QualityQuery(quality_criteria=criteria)
    expected = query.get_table_mask(table)
    expected_counts = query.to_table()["counts"]

    query = QualityQuery(quality_criteria=criteria)
    table_with_cols = query.add_table_mask_columns(table)

    assert table_with_cols["cut"].dtype == bool
    assert np.all(table_with_cols["cut"] == expected)
    assert np.all(query.to_table()["counts"] == expected_counts)


def test_add_columns_criteria_are_independent():
    """Test that a criterion cannot use the result of an earlier criterion."""

    query = QualityQuery(
        quality_criteria=[("low", "x > 0"), ("both", "low & (x < 10)")]
    )
    table = Table({"x": [-10, 1, 11]})

    with pytest.raises(ExpressionError):
        query.add_table_mask_columns(table)


@pytest.mark.parametrize(
    "criteria",
    [
        pytest.param([("x", "x > 0"), ("high", "x < 10")], id="existing_column"),
        pytest.param([("cut", "x > 0"), ("cut", "x < 10")], id="repeated_criterion"),
    ],
)
def test_add_columns_name_collision(criteria):
    """Test that criteria names may not collide when adding columns."""

    query = QualityQuery(quality_criteria=criteria)
    table = Table({"x": [-10, 1, 11]})

    with pytest.raises(QualityCriteriaError, match="not unique"):
        query.add_table_mask_columns(table)

    # the input and the statistics are unchanged
    assert table.colnames == ["x"]
    assert list(table["x"]) == [-10, 1, 11]
    assert np.all(query.to_table()["counts"] == 0)


@pytest.mark.parametrize("table_cls", [Table, QTable])
def test_add_columns_shallow_copy(table_cls):
    """Test that adding columns shares the data and leaves the input unchanged."""

    query = QualityQuery(quality_criteria=[("low", "x > 0")])
    table = table_cls(
        {
            "x": np.array([-10.0, 1.0, 11.0]),
            "time": Time([1, 2, 3], format="unix"),
        }
    )
    table.meta["key"] = "value"

    table_with_cols = query.add_table_mask_columns(table)

    assert type(table_with_cols) is table_cls
    assert table.colnames == ["x", "time"]
    assert table_with_cols.colnames == ["x", "time", "low"]
    assert table_with_cols.meta == table.meta
    assert np.shares_memory(table["x"], table_with_cols["x"])
    assert np.shares_memory(table["time"].jd1, table_with_cols["time"].jd1)
