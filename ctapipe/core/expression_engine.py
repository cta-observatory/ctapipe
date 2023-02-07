"""
Expression Engine
"""
import astropy.units as u  # for use in selection functions
import numpy as np  # for use in selection functions

# the following are what are allowed to be used
# in selection functions (passed to eval())
ALLOWED_GLOBALS = {"u": u, "np": np}  # astropy units  # numpy

for func in ("sin", "cos", "tan", "arctan2", "log", "log10", "exp", "sqrt"):
    ALLOWED_GLOBALS[func] = getattr(np, func)


class ExpressionError(TypeError):
    """Signal a problem with a user-defined selection criteria function"""


class ExpressionEngine:
    """
    Compile expressions on init, evaluate on call.
    """

    def __init__(self, expressions):
        self.expressions = expressions
        self._compile()

    def _compile(self):
        self.compiled = []
        for name, expression in self.expressions:
            if "lambda" in expression:
                raise ValueError(
                    "As of ctapipe 0.16, do not give lambda expressions,"
                    " instead directly give the expression."
                    " E.g. instead of `lambda p: p.hillas.width.value > 0`"
                    " use `parameters.hillas.width.value > 0`"
                )
            try:
                self.compiled.append(compile(expression, __name__, mode="eval"))
            except Exception as err:
                raise ExpressionError(
                    f"Error compiling expression '{expression}' for {name}\n"
                    f"{type(err).__name__}: {err}"
                ) from err

    def __call__(self, locals):
        for expr in self.compiled:
            try:
                yield eval(expr, ALLOWED_GLOBALS, locals)
            except NameError as err:
                raise ExpressionError(f"Error evaluating expression '{expr}': {err}")
            except Exception as err:
                raise ExpressionError(
                    f"Error evaluating expression '{expr}': {err}"
                ) from err

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["compiled"]
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._compile()
