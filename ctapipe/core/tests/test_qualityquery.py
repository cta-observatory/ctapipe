""" Tests of Selectors """
import numpy as np
import pytest
from astropy.table import Table

from ctapipe.core.expression_engine import ExpressionError
from ctapipe.core.qualityquery import QualityQuery
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
