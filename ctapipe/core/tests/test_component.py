import pytest
from traitlets import Float, TraitError

from .. import Component


def test_component_simple():
    """
    very basic test to construct a component and check
    that it's traits work correctly
    """

    class ExampleComponent(Component):
        description = "this is a test"
        param = Float(default_value=1.0,
                      help="float parameter").tag(config=True)

    comp = ExampleComponent(None)

    assert comp.has_trait('param') is True
    comp.param = 1.2

    with pytest.raises(TraitError):
        comp.param = "badvalue"
