import pickle

import pytest

from ctapipe.core.expression_engine import ExpressionEngine, ExpressionError


def test_failing_expression():
    """Test for invalid syntax in expression"""
    expressions = [("syntax-error", "log(a")]

    with pytest.raises(ExpressionError) as err:
        ExpressionEngine(expressions=expressions)

    assert "SyntaxError" in err.exconly()


def test_pickle():
    """Test for ExpressionEngine can be pickled"""
    expressions = [("foo", "5 * x"), ("bar", "10 * y")]
    engine = ExpressionEngine(expressions=expressions)
    data = pickle.dumps(engine)
    loaded = pickle.loads(data)

    assert loaded.expressions == expressions
    assert tuple(loaded({"x": 3, "y": 5})) == (15, 50)
