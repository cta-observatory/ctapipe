""" Tests of Selectors """
import numpy as np
import pytest
from astropy.table import Table

from ctapipe.core.expression_engine import ExpressionError
from ctapipe.core.qualityquery import QualityQuery
from ctapipe.core.traits import List


def test_selector(subarray_prod5_paranal):
    """test the functionality of an example Selector subclass"""

    class ExampleQualityQuery(QualityQuery):
        """Available variables: x"""

        quality_criteria = List(
            [
                ("high_enough", [("type", "*", "x > 3")]),
                ("a_value_not_too_high", [("type", "*", "x < 100")]),
                ("smallish", [("type", "*", "x < np.sqrt(100)")]),
            ],
        ).tag(config=True)

    query = ExampleQualityQuery(subarray=subarray_prod5_paranal)

    criteria1 = query(tel_id=1, x=0)  # pass smallish
    assert len(criteria1) == 3
    assert (criteria1 == [False, True, True]).all()

    criteria2 = query(tel_id=1, x=20)  # pass high_enough + not_too_high
    assert (criteria2 == [True, True, False]).all()

    criteria3 = query(tel_id=1, x=200)  # pass high_enough, fail not_too_high
    assert (criteria3 == [True, False, False]).all()

    criteria4 = query(tel_id=1, x=8)  # pass all
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


def test_invalid_input(subarray_prod5_paranal):
    class ExampleQualityQuery(QualityQuery):
        """Available variables: x"""

        quality_criteria = List(
            [
                ("high_enough", [("type", "*", "x > 3")]),
                ("a_value_not_too_high", [("type", "*", "x < 100")]),
                ("smallish", [("type", "*", "x < np.sqrt(100)")]),
            ],
        ).tag(config=True)

    query = ExampleQualityQuery(subarray=subarray_prod5_paranal)
    with pytest.raises(ExpressionError):
        query(tel_id=1, y=5)


def test_bad_selector(subarray_prod5_paranal):
    """ensure failure if a selector function is not a function or can't be evaluated"""

    query = QualityQuery(
        quality_criteria=[
            ("high_enough", [("type", "*", "x > 3")]),
            ("not good", [("type", "*", "foo")]),
            ("smallish", [("type", "*", "x < np.sqrt(100)")]),
        ],
        subarray=subarray_prod5_paranal,
    )
    with pytest.raises(ExpressionError):
        query(tel_id=1, x=5)

    # ensure we can't run arbitrary code.
    # try to construct something that is not in the
    # ALLOWED_GLOBALS list, but which is imported in selector.py
    # and see if it works in a function
    with pytest.raises(ExpressionError):
        query = QualityQuery(
            quality_criteria=[("dangerous", [("type", "*", "Component()")])],
            subarray=subarray_prod5_paranal,
        )
        query(tel_id=1, x=10)

    # test we only support expressions, not statements
    with pytest.raises(ExpressionError):
        query = QualityQuery(
            quality_criteria=[
                ("dangerous", [("type", "*", "import numpy; np.array([])")])
            ],
            subarray=subarray_prod5_paranal,
        )


def test_table_mask_and_to_table(subarray_prod5_paranal):
    """Test getting a mask for a whole table"""
    query = QualityQuery(
        quality_criteria=[
            ("inside unit circle", [("type", "*", "(x**2 + y**2) < 1.0")]),
            ("less than half", [("type", "*", "x < 0.5")]),
        ],
        subarray=subarray_prod5_paranal,
    )

    table = Table(
        {
            "tel_id": [1, 2, 3, 4, 5],
            "x": [1.0, 0.2, -0.5, 0.6, 0.7],
            "y": [0.0, 0.5, 1.0, 0.2, 0.1],
        }
    )

    mask = query.get_table_mask(table)
    assert len(mask) == len(table)
    assert mask.dtype == np.bool_
    np.testing.assert_equal(mask, [False, True, False, False, False])
    stats = query.to_table()
    np.testing.assert_equal(stats["counts"], [5, 3, 2])
    np.testing.assert_equal(stats["cumulative_counts"], [5, 3, 1])


def test_to_table_after_call(subarray_prod5_paranal):
    """Test counting and conversion to table."""
    query = QualityQuery(
        quality_criteria=[
            ("inside unit circle", [("type", "*", "(x**2 + y**2) < 1.0")]),
            ("less than half", [("type", "*", "x < 0.5")]),
        ],
        subarray=subarray_prod5_paranal,
    )
    assert np.all(query(tel_id=1, x=0.3, y=0))
    assert not np.all(query(tel_id=1, x=1, y=0))

    assert np.all(query(tel_id=2, x=0.3, y=0))
    assert not np.all(query(tel_id=2, x=1, y=0))

    stats = query.to_table()
    np.testing.assert_equal(stats["counts"], [4, 2, 2])
    np.testing.assert_equal(stats["cumulative_counts"], [4, 2, 2])


def test_printing(subarray_prod5_paranal):
    """Just check the query can be stringified correctly"""
    query = QualityQuery(
        quality_criteria=[("check", [("type", "*", "x>3")])],
        subarray=subarray_prod5_paranal,
    )

    assert isinstance(str(query), str)


def test_setup(subarray_prod5_paranal):
    """Test that tuples can only have length 2"""
    QualityQuery(
        quality_criteria=[
            ("foo", [("type", "*", "expression")]),
            ("bar", [("type", "*", "expression"), ("id", "1", "another-expresssion")]),
            ("baz", "expression"),
        ],
        subarray=subarray_prod5_paranal,
    )


def test_with_lambda(subarray_prod5_paranal):
    """Test that we raise an error when using lambda expresssions,
    which we don't support since v0.16.0"""

    with pytest.raises(ValueError, match="lambda"):
        QualityQuery(
            quality_criteria=[("name", [("type", "*", "lambda p: p > 2")])],
            subarray=subarray_prod5_paranal,
        )


def test_telescope_component(subarray_prod5_paranal):
    """Test that more specific patterns override more general."""
    query = QualityQuery(
        quality_criteria=[
            ("foo", [("type", "*", "x > 0"), ("id", "2", "x < 0")]),
        ],
        subarray=subarray_prod5_paranal,
    )

    x = 1
    assert query(x=x, tel_id=1)
    assert not query(x=x, tel_id=2)
