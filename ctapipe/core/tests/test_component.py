from abc import abstractmethod, ABC
import pytest
from traitlets import Float, TraitError
from traitlets.config.loader import Config
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
    extra = Float(default_value=5.0,
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


def test_config():
    config = Config()
    config['ExampleComponent'] = Config()
    config['ExampleComponent']['param'] = 199.
    comp = ExampleComponent(config=config)
    assert comp.param == 199.


def test_config_baseclass():
    config = Config()
    config['ExampleComponent'] = Config()
    config['ExampleComponent']['param'] = 199.
    comp1 = ExampleSubclass1(config=config)
    assert comp1.param == 199.
    comp2 = ExampleSubclass2(config=config)
    assert comp2.param == 199.


def test_config_subclass1():
    config = Config()
    config['ExampleSubclass1'] = Config()
    config['ExampleSubclass1']['param'] = 199.
    comp = ExampleComponent(config=config)
    assert comp.param == 1.


def test_config_subclass2():
    config = Config()
    config['ExampleSubclass2'] = Config()
    config['ExampleSubclass2']['param'] = 199.
    comp = ExampleComponent(config=config)
    assert comp.param == 1.


def test_config_sibling1():
    config = Config()
    config['ExampleSubclass1'] = Config()
    config['ExampleSubclass1']['param'] = 199.
    comp1 = ExampleSubclass1(config=config)
    assert comp1.param == 199.
    comp2 = ExampleSubclass2(config=config)
    assert comp2.param == 3.


def test_config_sibling2():
    config = Config()
    config['ExampleSubclass2'] = Config()
    config['ExampleSubclass2']['param'] = 199.
    comp1 = ExampleSubclass1(config=config)
    assert comp1.param == 1.
    comp2 = ExampleSubclass2(config=config)
    assert comp2.param == 199.


def test_config_baseclass_then_subclass():
    config = Config()
    config['ExampleComponent'] = Config()
    config['ExampleComponent']['param'] = 199.
    config['ExampleSubclass1'] = Config()
    config['ExampleSubclass1']['param'] = 229.
    comp = ExampleSubclass1(config=config)
    assert comp.param == 229.


def test_config_subclass_then_baseclass():
    config = Config()
    config['ExampleSubclass1'] = Config()
    config['ExampleSubclass1']['param'] = 229.
    config['ExampleComponent'] = Config()
    config['ExampleComponent']['param'] = 199.
    comp = ExampleSubclass1(config=config)
    assert comp.param == 229.


def test_config_override():
    config = Config()
    config['ExampleComponent'] = Config()
    config['ExampleComponent']['param'] = 199.
    comp = ExampleComponent(config=config, param=229.)
    assert comp.param == 229.


def test_config_override_subclass():
    config = Config()
    config['ExampleComponent'] = Config()
    config['ExampleComponent']['param'] = 199.
    comp = ExampleSubclass1(config=config, param=229.)
    assert comp.param == 229.


def test_extra():
    comp = ExampleSubclass2(extra=229.)
    assert comp.has_trait('extra') is True
    assert comp.extra == 229.


def test_extra_config():
    config = Config()
    config['ExampleSubclass2'] = Config()
    config['ExampleSubclass2']['extra'] = 229.
    comp = ExampleSubclass2(config=config)
    assert comp.extra == 229.


def test_extra_missing():
    with pytest.raises(TraitError):
        ExampleSubclass1(extra=229.)


def test_extra_config_missing():
    config = Config()
    config['ExampleSubclass1'] = Config()
    config['ExampleSubclass1']['extra'] = 199.
    with pytest.warns(UserWarning):
        comp = ExampleSubclass1(config=config)
    assert comp.has_trait('extra') is False
    with pytest.raises(AttributeError):
        assert comp.extra == 229.


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
    assert comp.param == 3.  # No change as it is a seperate traitlet object
    ExampleComponent.param.default_value = old_default


def test_help_changed_default():
    old_default = ExampleComponent.param.default_value
    ExampleComponent.param.default_value = 199.
    help_msg = ExampleComponent.class_get_help()
    assert "Default: 199.0" in help_msg
    ExampleComponent.param.default_value = old_default


def test_from_name():
    subclass = ExampleComponent.from_name("ExampleSubclass1")
    assert isinstance(subclass, ExampleSubclass1)
    subclass = ExampleComponent.from_name("ExampleSubclass2")
    assert isinstance(subclass, ExampleSubclass2)


def test_from_name_config():
    config = Config({'ExampleComponent': {'param': 229.}})
    subclass = ExampleComponent.from_name("ExampleSubclass1", config=config)
    assert subclass.param == 229.
