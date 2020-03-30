""" Tests of Selectors """
import pytest

from ctapipe.core.qualityquery import QualityQuery, QualityCriteriaError
from ctapipe.core.traits import List


def test_selector():
    """ test the functionality of an example Selector subclass"""

    class ExampleQualityQuery(QualityQuery):
        quality_criteria = List(
            default_value=[
                ("high_enough", "lambda x: x > 3"),
                ("a_value_not_too_high", "lambda x: x < 100"),
                ("smallish", "lambda x: x < np.sqrt(100)"),
            ],
        ).tag(config=True)

    query = ExampleQualityQuery()

    criteria1 = query(0)  # pass smallish
    assert len(criteria1) == 3
    assert (criteria1 == [False, True, True]).all()

    criteria2 = query(20)  # pass high_enough + not_too_high
    assert (criteria2 == [True, True, False]).all()

    criteria3 = query(200)  # pass high_enough, fail not_too_high
    assert (criteria3 == [True, False, False]).all()

    criteria4 = query(8)  # pass all
    assert (criteria4 == True).all()

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

    # check that the order is preserved
    assert query.criteria_names[1] == "high_enough"
    assert query.criteria_names[2] == "a_value_not_too_high"
    assert query.criteria_names[3] == "smallish"

    # check we can get back the correct function string:
    assert query.selection_function_strings[1] == "lambda x: x > 3"

    assert len(query) == 4  # 4 events counted


def test_bad_selector():
    """ ensure failure if a selector function is not a function or can't be evaluated"""

    with pytest.raises(QualityCriteriaError):
        s = QualityQuery(
            quality_criteria=[
                ("high_enough", "lambda x: x > 3"),
                ("not_good", "3"),
                ("smallish", "lambda x: x < 10"),
            ]
        )
        assert s

    with pytest.raises(QualityCriteriaError):
        s = QualityQuery(
            quality_criteria=[
                ("high_enough", "lambda x: x > 3"),
                ("not_good", "x == 3"),
                ("smallish", "lambda x: x < 10"),
            ]
        )
        assert s

    # ensure we can't run arbitrary code.
    # try to construct something that is not in the
    # ALLOWED_GLOBALS list, but which is imported in selector.py
    # and see if it works in a function
    with pytest.raises(NameError):
        s = QualityQuery(quality_criteria=[("dangerous", "lambda x: Component()")])
        s(10)
