import warnings
from abc import ABC, abstractmethod

import pytest
from traitlets import Float, Int, TraitError
from traitlets.config.loader import Config

from ctapipe.core import Component, Tool


def test_non_abstract_children():
    """check that we can find all constructable children"""
    from ctapipe.core import non_abstract_children

    class AbstractBase(ABC):
        @abstractmethod
        def method(self):
            pass

    class Child1(AbstractBase):
        def method(self):
            print("method of Child1")

    class Child2(AbstractBase):
        def method(self):
            print("method of Child2")

    class GrandChild(Child2):
        def method(self):
            print("method of GrandChild")

    class AbstractChild(AbstractBase):
        pass

    kids = non_abstract_children(AbstractBase)
    assert Child1 in kids
    assert Child2 in kids
    assert GrandChild in kids
    assert AbstractChild not in kids


def test_get_config_from_hierarchy():
    from ctapipe.core.component import find_config_in_hierarchy

    class Bottom(Component):
        val = Float(0).tag(config=True)

    class Middle(Component):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.bottom = Bottom(parent=self)

    class Top(Component):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.middle = Middle(parent=self)

    # test with root present
    c = Config({"Top": {"Middle": {"Bottom": {"val": 5}}}})
    t = Top(config=c)

    val = find_config_in_hierarchy(t.middle, "Bottom", "val")
    assert val == 5

    # test with root not present
    c = Config({"Middle": {"Bottom": {"val": 5}}})
    t = Top(config=c)

    val = find_config_in_hierarchy(t.middle, "Bottom", "val")
    assert val == 5


class SubComponent(Component):
    """An Example Component, this is the help text"""

    value = Float(default_value=-1.0, help="float parameter").tag(config=True)


class ExampleComponent(Component):
    """An Example Component, this is the help text"""

    param = Float(default_value=1.0, help="float parameter").tag(config=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sub = SubComponent(parent=self)


class ExampleSubclass1(ExampleComponent):
    """a subclass of ExampleComponent"""

    pass


class ExampleSubclass2(ExampleComponent):
    """Another ExampleComponent"""

    description = "A shorter description"
    param = Float(default_value=3.0, help="float parameter").tag(config=True)
    extra = Float(default_value=5.0, help="float parameter").tag(config=True)


def test_component_is_abstract():
    """check that we can make an abstract component"""

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

    assert comp.has_trait("param") is True
    comp.param = 1.2

    with pytest.raises(TraitError):
        comp.param = "badvalue"


def test_component_kwarg_setting():
    """check that we can construct a component by setting traits via kwargs"""
    comp = ExampleComponent(param=3)
    assert comp.param == 3

    # Invalid type
    with pytest.raises(TraitError):
        comp = ExampleComponent(param="badvalue")

    # Invalid traitlet
    with pytest.raises(TraitError):
        # the current traitlets version already warns about this
        # will be raising an error in the future, but we want the error
        # now, filter the warning here to not clutter the log
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            comp = ExampleComponent(incorrect="wrong")


def test_help():
    """check that component help strings are generated correctly"""
    help_msg = ExampleComponent.class_get_help()
    assert "Default: 1.0" in help_msg


def test_config():
    """check that components can be constructed by config dict"""
    config = Config()
    config["ExampleComponent"] = Config()
    config["ExampleComponent"]["param"] = 199.0
    comp = ExampleComponent(config=config)
    assert comp.param == 199.0


def test_config_baseclass():
    """check that parent and subclass configuration works"""
    config = Config()
    config["ExampleComponent"] = Config()
    config["ExampleComponent"]["param"] = 199.0
    comp1 = ExampleSubclass1(config=config)
    assert comp1.param == 199.0
    comp2 = ExampleSubclass2(config=config)
    assert comp2.param == 199.0


def test_config_subclass1():
    """check sub-class config"""
    config = Config()
    config["ExampleSubclass1"] = Config()
    config["ExampleSubclass1"]["param"] = 199.0
    comp = ExampleComponent(config=config)
    assert comp.param == 1.0


def test_config_subclass2():
    """check another sub-class config"""
    config = Config()
    config["ExampleSubclass2"] = Config()
    config["ExampleSubclass2"]["param"] = 199.0
    comp = ExampleComponent(config=config)
    assert comp.param == 1.0


def test_config_sibling1():
    """check sibling config"""
    config = Config()
    config["ExampleSubclass1"] = Config()
    config["ExampleSubclass1"]["param"] = 199.0
    comp1 = ExampleSubclass1(config=config)
    assert comp1.param == 199.0
    comp2 = ExampleSubclass2(config=config)
    assert comp2.param == 3.0


def test_config_sibling2():
    """check sibling config"""
    config = Config()
    config["ExampleSubclass2"] = Config()
    config["ExampleSubclass2"]["param"] = 199.0
    comp1 = ExampleSubclass1(config=config)
    assert comp1.param == 1.0
    comp2 = ExampleSubclass2(config=config)
    assert comp2.param == 199.0


def test_config_baseclass_then_subclass():
    """check base and subclass config"""
    config = Config()
    config["ExampleComponent"] = Config()
    config["ExampleComponent"]["param"] = 199.0
    config["ExampleSubclass1"] = Config()
    config["ExampleSubclass1"]["param"] = 229.0
    comp = ExampleSubclass1(config=config)
    assert comp.param == 229.0


def test_config_subclass_then_baseclass():
    """check subclass and base config"""
    config = Config()
    config["ExampleSubclass1"] = Config()
    config["ExampleSubclass1"]["param"] = 229.0
    config["ExampleComponent"] = Config()
    config["ExampleComponent"]["param"] = 199.0
    comp = ExampleSubclass1(config=config)
    assert comp.param == 229.0


def test_config_override():
    """check that we can override a trait set in the config"""
    config = Config()
    config["ExampleComponent"] = Config()
    config["ExampleComponent"]["param"] = 199.0
    comp = ExampleComponent(config=config, param=229.0)
    assert comp.param == 229.0


def test_config_override_subclass():
    """check that we can override a trait set in the config"""
    config = Config()
    config["ExampleComponent"] = Config()
    config["ExampleComponent"]["param"] = 199.0
    comp = ExampleSubclass1(config=config, param=229.0)
    assert comp.param == 229.0


def test_extra():
    """check that traits are settable"""
    comp = ExampleSubclass2(extra=229.0)
    assert comp.has_trait("extra") is True
    assert comp.extra == 229.0


def test_extra_config():
    """check setting trait via config"""
    config = Config()
    config["ExampleSubclass2"] = Config()
    config["ExampleSubclass2"]["extra"] = 229.0
    comp = ExampleSubclass2(config=config)
    assert comp.extra == 229.0


def test_unknown_traitlet_raises():
    """check that setting an incorrect trait raises an exception"""
    with pytest.raises(TraitError):
        # the current traitlets version already warns about this
        # will be raising an error in the future, but we want the error
        # now, filter the warning here to not clutter the log
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)

            ExampleSubclass1(extra=229.0)


def test_extra_config_missing():
    """
    check that setting an incorrect trait via config also raises
    an exception
    """
    config = Config()
    config["ExampleSubclass1"] = Config()
    config["ExampleSubclass1"]["extra"] = 199.0

    with pytest.raises(TraitError):
        ExampleSubclass1(config=config)


def test_default():
    """check default values work"""
    comp = ExampleComponent()
    assert comp.param == 1.0


def test_default_subclass():
    """check default values work in subclasses"""
    comp = ExampleSubclass1()
    assert comp.param == 1.0


def test_default_subclass_override():
    """check overrides work in subclasses"""
    comp = ExampleSubclass2()
    assert comp.param == 3.0


def test_change_default():
    """check we can change a default value"""
    old_default = ExampleComponent.param.default_value
    ExampleComponent.param.default_value = 199.0
    comp = ExampleComponent()
    assert comp.param == 199.0
    ExampleComponent.param.default_value = old_default


def test_change_default_subclass():
    """check we can change a default value in subclass"""
    old_default = ExampleComponent.param.default_value
    ExampleComponent.param.default_value = 199.0
    comp = ExampleSubclass1()
    assert comp.param == 199.0
    ExampleComponent.param.default_value = old_default


def test_change_default_subclass_override():
    """check override default value"""
    old_default = ExampleComponent.param.default_value
    ExampleComponent.param.default_value = 199.0
    comp = ExampleSubclass2()
    assert comp.param == 3.0  # No change as it is a separate traitlet object
    ExampleComponent.param.default_value = old_default


def test_help_changed_default():
    """check that the help text is updated if the default is changed"""
    old_default = ExampleComponent.param.default_value
    ExampleComponent.param.default_value = 199.0
    help_msg = ExampleComponent.class_get_help()
    assert "Default: 199.0" in help_msg
    ExampleComponent.param.default_value = old_default


def test_non_abstract_subclasses():
    """non_abstract_subclasses() is a helper function:
    it might just be nice, in case a person wants to see the subclasses
    in a python session.

    Can also be helpful in parametrized tests, to make sure all
    subclasses are being tested.
    """
    assert "ExampleSubclass1" in ExampleComponent.non_abstract_subclasses()


def test_from_name():
    """Make sure one can construct a Component subclass by name"""
    subclass = ExampleComponent.from_name("ExampleSubclass1")
    assert isinstance(subclass, ExampleSubclass1)
    subclass = ExampleComponent.from_name("ExampleSubclass2")
    assert isinstance(subclass, ExampleSubclass2)


def test_from_name_config():
    """make sure one can construct a Component subclass by name + config"""
    config = Config({"ExampleComponent": {"param": 229.0}})
    subclass = ExampleComponent.from_name("ExampleSubclass1", config=config)
    assert subclass.param == 229.0


def test_component_current_config():
    """make sure one can get the full current configuration"""
    comp = ExampleComponent()
    full_config = comp.get_current_config()
    assert "ExampleComponent" in full_config
    assert "param" in full_config["ExampleComponent"]
    assert full_config["ExampleComponent"]["param"] == 1.0


def test_component_html_repr():
    """check the HTML repr for Jupyter notebooks"""
    comp = ExampleComponent()
    html = comp._repr_html_()
    assert len(html) > 10


def test_telescope_component(subarray_prod5_paranal):
    from ctapipe.core import TelescopeComponent

    class Base(TelescopeComponent):
        pass

    class Sub(Base):
        pass

    assert isinstance(Base.from_name("Sub", subarray=subarray_prod5_paranal), Sub)


def test_full_config():
    class SubComponent(Component):
        param = Int(default_value=3).tag(config=True)

    class MyComponent(Component):
        val = Int(default_value=42).tag(config=True)

        def __init__(self, config=None, parent=None):
            super().__init__(config=config, parent=parent)
            self.sub = SubComponent(parent=self)

    comp = MyComponent()
    assert comp.get_current_config() == {
        "MyComponent": {"val": 42, "SubComponent": {"param": 3}}
    }

    # test round tripping
    comp = MyComponent()
    comp.val = 10
    comp.sub.param = -1

    dict_config = comp.get_current_config()
    config = Config(dict_config)
    comp_from_config = MyComponent(config=config)

    assert dict_config == comp_from_config.get_current_config()


def test_logging_hierarchy():
    class Foo(Component):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    class Bar(Component):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.foo = Foo(parent=self)

    class Baz(Tool):
        name = "baz"

        def setup(self):
            self.bar = Bar(parent=self)

        def start(self):
            pass

    baz = Baz()
    baz.setup()

    assert baz.log.name == "ctapipe.baz"
    assert baz.bar.log.name == "ctapipe.baz.Bar"
    assert baz.bar.foo.log.name == "ctapipe.baz.Bar.Foo"
