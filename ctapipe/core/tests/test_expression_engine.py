import pytest

from ctapipe.core.expression_engine import ExpressionEngine, ExpressionError


def test_failing_expression():
    expressions = [("syntax-error", "log(a")]

    with pytest.raises(ExpressionError) as err:
        ExpressionEngine(expressions=expressions)

    assert "SyntaxError" in err.exconly()
