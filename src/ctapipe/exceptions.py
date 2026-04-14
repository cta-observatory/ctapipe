__all__ = [
    "CTAPipeException",
    "TooFewEvents",
    "OptionalDependencyMissing",
    "InputMissing",
    "MockOptionalDecorator",
]


class CTAPipeException(Exception):
    pass


class TooFewEvents(CTAPipeException):
    """Raised if something that needs a minimum number of event gets fewer"""


class OptionalDependencyMissing(ModuleNotFoundError):
    """Raised if an optional dependency required for a feature is not installed"""

    def __init__(self, module):
        self.module = module
        msg = f"'{module}' is required for this functionality of ctapipe"
        super().__init__(msg)


class InputMissing(ValueError):
    """Raised in case an input was not specified."""


class MockOptionalDecorator:
    """A decorator that can be used in-place of an imported decorator.

    Will throw the corresponding OptionalDependencyMissing exception when
    the decorated function is called.

    Examples
    --------
    You might want to use this class for optional dependencies that provide
    decorators. With decorators, it is hard to defer import of the dependency
    to runtime. See this example of how one might make numba with njit optional:

        from unittest import MagicMock
        from ctapipe.exceptions import MockOptionalDecorator
        try:
            import numba
        except ModuleNotFoundError:
            numba = MagicMock()
            numba.njit = MockOptionalDecorator("numba")

        @numba.njit(cache=True)
        def example(x):
            return 5 * x
    """

    def __init__(self, module):
        self.module = module

    def __call__(self, *args, **kwargs):
        def _raise(*args, **kwargs):
            raise OptionalDependencyMissing(self.module)

        # decorator called as @decorator without arguments
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return _raise

        # decorator called as @decorator(*args, **kwargs)
        def wrapper(func):
            return _raise

        return wrapper
