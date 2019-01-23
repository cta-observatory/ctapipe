from abc import abstractmethod, ABC
import pytest
from traitlets import Float, TraitError

from ctapipe.core import Component


def test_non_abstract_children():
    from ctapipe.core import non_abstract_children

    class AbstractBase(ABC):
        @abstractmethod
        def method(self):
            pass

    class Child1(AbstractBase):
        def method(self):
            print('method of Child1')

    class Child2(AbstractBase):
        def method(self):
            print('method of Child2')

    class GrandChild(Child2):
        def method(self):
            print('method of GrandChild')

    class AbstractChild(AbstractBase):
        pass

    kids = non_abstract_children(AbstractBase)
    assert Child1 in kids
    assert Child2 in kids
    assert GrandChild in kids
    assert AbstractChild not in kids


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

    class ExampleComponent(Component):
        description = "this is a test"
        param = Float(default_value=1.0,
                      help="float parameter").tag(config=True)

    comp = ExampleComponent()

    assert comp.has_trait('param') is True
    comp.param = 1.2

    with pytest.raises(TraitError):
        comp.param = "badvalue"


def test_component_kwarg_setting():
    class ExampleComponent(Component):
        description = "this is a test"
        param = Float(default_value=1.0,
                      help="float parameter").tag(config=True)

    comp = ExampleComponent(param=3)
    assert comp.param == 3

    # Invalid type
    with pytest.raises(TraitError):
        comp = ExampleComponent(param="badvalue")

    # Invalid traitlet
    with pytest.raises(TraitError):
        comp = ExampleComponent(incorrect="wrong")
