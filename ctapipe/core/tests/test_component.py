import pytest
from traitlets import Float, TraitError

from ctapipe.core import Component
from abc import abstractmethod


class ExampleComponent(Component):
    description = "this is a test"
    param = Float(default_value=1.0,
                  help="float parameter").tag(config=True)


class ExampleSubclass1(ExampleComponent):
    description = "this is a test"


class ExampleSubclass2(ExampleComponent):
    description = "this is a test"
    param = Float(default_value=3.0,
                  help="float parameter").tag(config=True)


def test_component_is_abstract():

    class AbstractComponent(Component):
        @abstractmethod
        def test(self):
            pass

    with pytest.raises(TypeError):
        AbstractComponent()


def test_component_simple():
    """
    very basic test to construct a component and check
    that it's traits work correctly
    """
    comp = ExampleComponent()

    assert comp.has_trait('param') is True
    comp.param = 1.2

    with pytest.raises(TraitError):
        comp.param = "badvalue"


def test_component_kwarg_setting():

    comp = ExampleComponent(param=3)
    assert comp.param == 3

    # Invalid type
    with pytest.raises(TraitError):
        comp = ExampleComponent(param="badvalue")

    # Invalid traitlet
    with pytest.raises(TraitError):
        comp = ExampleComponent(incorrect="wrong")


def test_help():
    help_msg = ExampleComponent.class_get_help()
    assert "Default: 1.0" in help_msg


def test_default():
    comp = ExampleComponent()
    assert comp.param == 1.


def test_default_subclass():
    comp = ExampleSubclass1()
    assert comp.param == 1.


def test_default_subclass_override():
    comp = ExampleSubclass2()
    assert comp.param == 3.


def test_change_default():
    old_default = ExampleComponent.param.default_value
    ExampleComponent.param.default_value = 199.
    comp = ExampleComponent()
    assert comp.param == 199.
    ExampleComponent.param.default_value = old_default


def test_change_default_subclass():
    old_default = ExampleComponent.param.default_value
    ExampleComponent.param.default_value = 199.
    comp = ExampleSubclass1()
    assert comp.param == 199.
    ExampleComponent.param.default_value = old_default


def test_change_default_subclass_override():
    old_default = ExampleComponent.param.default_value
    ExampleComponent.param.default_value = 199.
    comp = ExampleSubclass2()
    assert comp.param == 199.
    ExampleComponent.param.default_value = old_default


def test_help_changed_default():
    old_default = ExampleComponent.param.default_value
    ExampleComponent.param.default_value = 199.
    help_msg = ExampleComponent.class_get_help()
    assert "Default: 199.0" in help_msg
    ExampleComponent.param.default_value = old_default
