from unittest.mock import MagicMock

import pytest


def test_optional_decorator_no_args():
    from ctapipe.exceptions import MockOptionalDecorator, OptionalDependencyMissing

    dummy_opt_module = MagicMock()
    dummy_opt_module.some_decorator = MockOptionalDecorator("dummy")

    # defining the function should not throw
    @dummy_opt_module.some_decorator
    def func():
        pass

    with pytest.raises(OptionalDependencyMissing, match="'dummy' is required"):
        # calling the function should raise
        func()


def test_optional_decorator_with_args():
    from ctapipe.exceptions import MockOptionalDecorator, OptionalDependencyMissing

    dummy_opt_module = MagicMock()
    dummy_opt_module.some_decorator = MockOptionalDecorator("dummy")

    # defining the function should not throw
    @dummy_opt_module.some_decorator(foo=5)
    def func():
        pass

    with pytest.raises(OptionalDependencyMissing, match="'dummy' is required"):
        # calling the function should raise
        func()
