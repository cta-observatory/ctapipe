""" Tests of Selectors """
import pytest

from ctapipe.core.selector import Selector, SelectionFunctionError
from ctapipe.core.traits import Dict


def test_selector():
    """ test the functionality of an example Selector subclass"""

    class ExampleSelector(Selector):
        selection_functions = Dict(
            default_value=dict(
                high_enough="lambda x: x > 3",
                a_value_not_too_high="lambda x: x < 100",
                smallish="lambda x: x < 10",
            ),
        ).tag(config=True)

    select = ExampleSelector()

    criteria1 = select(0)  # pass smallish
    assert len(criteria1) == 3
    assert (criteria1 == [False, True, True]).all()

    criteria2 = select(20)  # pass high_enough + not_too_high
    assert (criteria2 == [True, True, False]).all()

    criteria3 = select(200)  # pass high_enough, fail not_too_high
    assert (criteria3 == [True, False, False]).all()

    criteria4 = select(8)  # pass all
    assert (criteria4 == True).all()

    tab = select.to_table()
    html = select._repr_html_()
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
    assert select.criteria_names[1] == "high_enough"
    assert select.criteria_names[2] == "a_value_not_too_high"
    assert select.criteria_names[3] == "smallish"

    # check we can get back the correct function string:
    assert select.selection_function_strings[1] == "lambda x: x > 3"

    assert len(select) == 4  # 4 events counted

def test_bad_selector():
    """ ensure failure if a selector function is not a function or can't be evaluated"""

    with pytest.raises(SelectionFunctionError):
        s = Selector(
            selection_functions=dict(
                high_enough="lambda x: x > 3",
                not_good="3",
                smallish="lambda x: x < 10",
            )
        )
        assert s

    with pytest.raises(SelectionFunctionError):
        s = Selector(
            selection_functions=dict(
                high_enough="lambda x: x > 3",
                not_good="x == 3",
                smallish="lambda x: x < 10",
            )
        )
        assert s

