"""
Expression Engine
"""
import astropy.units as u  # for use in selection functions
import numpy as np  # for use in selection functions

from .component import Component
from .traits import List, Tuple, Unicode

# the following are what are allowed to be used
# in selection functions (passed to eval())
ALLOWED_GLOBALS = {"u": u, "np": np}  # astropy units  # numpy

for func in ("sin", "cos", "tan", "arctan2", "log", "log10", "exp", "sqrt"):
    ALLOWED_GLOBALS[func] = getattr(np, func)


class ExpressionError(TypeError):
    """Signal a problem with a user-defined selection criteria function"""


class ExpressionEngine(Component):
    """
    Compile expressions on init, evaluate on call.
    """

    expressions = List(
        Tuple(Unicode(), Unicode()),
        help="List of 2-Tuples of Strings: ('name', 'expression').",
    ).tag(config=True)

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)

        self.compiled = []
        for name, expression in self.expressions:
            try:
                self.compiled.append(compile(expression, __name__, mode="eval"))
            except Exception:
                raise ExpressionError(
                    f"Error compiling expression '{expression}' for {name}"
                )

    def __call__(self, locals):
        for expr in self.compiled:
            try:
                yield eval(expr, ALLOWED_GLOBALS, locals)
            except NameError as err:
                raise ExpressionError(f"Error evaluating expression '{expr}': {err}")
            except Exception as err:
                raise ExpressionError(f"Error evaluating expression '{expr}'") from err
