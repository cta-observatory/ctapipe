from ctapipe.core.factory import Factory, child_subclasses
from ctapipe.core.component import Component
from traitlets import Int, TraitError
import pytest
from traitlets.config.loader import Config


class ExampleComponentParent(Component):
    value = Int(123, help="").tag(config=True)

    def __init__(self, config, parent, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)


class ExampleComponent1(ExampleComponentParent):
    pass


class ExampleComponent2(ExampleComponentParent):
    value = Int(123222, help="").tag(config=True)


class ExampleComponent3(ExampleComponentParent):
    value = Int(123333, help="").tag(config=True)

    def __init__(self, config, parent, extra=0, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)
        self.extra = extra


class ExampleComponent4(ExampleComponentParent):
    value = Int(123444, help="").tag(config=True)
    extra = Int(5, help="").tag(config=True)


class ExampleFactory(Factory):
    base = ExampleComponentParent
    default = 'ExampleComponent1'
    product_help = "Product for testing"


class SecondExampleFactory(Factory):
    base = ExampleComponentParent
    default = 'Should be different'
    product_help = "Should be different"


class IncorrectFactory(Factory):
    base = ExampleComponentParent
    default = 'ExampleComponent1'

    def _get_product_name(self):
        return "NonExistantClass"


def test_factory():
    obj = ExampleFactory(product='ExampleComponent2').get_product()
    assert(obj.__class__.__name__ == 'ExampleComponent2')
    assert(obj.value == 123222)


def test_factory_config():
    config = Config()
    config['ExampleComponent2'] = Config()
    config['ExampleComponent2']['value'] = 111
    obj = ExampleFactory(
        product='ExampleComponent2', config=config
    ).get_product()
    assert(obj.__class__.__name__ == 'ExampleComponent2')
    assert(obj.value == 111)


def test_factory_config_via_parent():
    config = Config()
    config['ExampleComponentParent'] = Config()
    config['ExampleComponentParent']['value'] = 111
    obj = ExampleFactory(
        product='ExampleComponent2', config=config
    ).get_product()
    assert(obj.__class__.__name__ == 'ExampleComponent2')
    assert(obj.value == 111)


def test_factory_config_via_sibling():
    config = Config()
    config['ExampleComponent1'] = Config()
    config['ExampleComponent1']['value'] = 111
    obj = ExampleFactory(
        product='ExampleComponent2', config=config
    ).get_product()
    assert(obj.__class__.__name__ == 'ExampleComponent2')
    with pytest.raises(AssertionError):
        assert(obj.value == 111)


def test_factory_config_extra():
    config = Config()
    config['ExampleComponent4'] = Config()
    config['ExampleComponent4']['value'] = 111
    config['ExampleComponent4']['extra'] = 112
    obj = ExampleFactory(
        product='ExampleComponent4', config=config
    ).get_product()
    assert(obj.__class__.__name__ == 'ExampleComponent4')
    assert(obj.value == 111)
    assert(obj.extra == 112)


def test_factory_config_via_parent_extra():
    config = Config()
    config['ExampleComponentParent'] = Config()
    config['ExampleComponentParent']['value'] = 111
    config['ExampleComponentParent']['extra'] = 112
    obj = ExampleFactory(
        product='ExampleComponent4', config=config
    ).get_product()
    assert(obj.__class__.__name__ == 'ExampleComponent4')
    assert(obj.value == 111)
    assert(obj.extra == 112)


def test_factory_config_via_sibling_extra():
    config = Config()
    config['ExampleComponent1'] = Config()
    config['ExampleComponent1']['value'] = 111
    config['ExampleComponent1']['extra'] = 112
    obj = ExampleFactory(
        product='ExampleComponent4', config=config
    ).get_product()
    assert(obj.__class__.__name__ == 'ExampleComponent4')
    with pytest.raises(AssertionError):
        assert(obj.value == 111)
    with pytest.raises(AssertionError):
        assert(obj.extra == 112)


def test_factory_traitlet_default():
    old_default = ExampleComponentParent.value.default_value
    ExampleComponentParent.value.default_value = 199
    assert ExampleComponent1.value.default_value == 199
    obj = ExampleFactory(product='ExampleComponent1').get_product()
    assert(obj.__class__.__name__ == 'ExampleComponent1')
    assert(obj.value == 199)
    ExampleComponentParent.value.default_value = old_default


def test_factory_traitlet_default_seperate_traitlet():
    """
    Possibly want to allow such functionality in the future!
    """
    old_default = ExampleComponentParent.value.default_value
    ExampleComponentParent.value.default_value = 199
    with pytest.raises(AssertionError):
        assert ExampleComponent2.value.default_value == 199
    obj = ExampleFactory(product='ExampleComponent2').get_product()
    assert(obj.__class__.__name__ == 'ExampleComponent2')
    with pytest.raises(AssertionError):
        assert(obj.value == 199)
    ExampleComponentParent.value.default_value = old_default


def test_factory_traitlet_default_config():
    old_default = ExampleComponentParent.value.default_value
    ExampleComponentParent.value.default_value = 199
    config = Config()
    config['ExampleComponentParent'] = Config()
    config['ExampleComponentParent']['value'] = 111
    obj = ExampleFactory(
        product='ExampleComponent1', config=config
    ).get_product()
    assert(obj.__class__.__name__ == 'ExampleComponent1')
    assert(obj.value == 111)
    ExampleComponentParent.value.default_value = old_default


def test_second_factory_product_different():
    ExampleFactory.update_product_traitlet()
    SecondExampleFactory.update_product_traitlet()
    assert ExampleFactory.product != SecondExampleFactory.product
    assert ExampleFactory.product.help != SecondExampleFactory.product.help


def test_factory_subclass_detection():
    subclasses = [
        ExampleComponent1,
        ExampleComponent2,
        ExampleComponent3,
        ExampleComponent4
    ]
    subclasses_str = [str(i) for i in subclasses]

    factory_subclasses = child_subclasses(ExampleFactory.base).values()
    factory_subclasses_str = [str(i) for i in factory_subclasses]

    subclass_names = [
        "ExampleComponent1",
        "ExampleComponent2",
        "ExampleComponent3",
        "ExampleComponent4"
    ]
    factory_subclass_names = child_subclasses(ExampleFactory.base).keys()

    ExampleFactory.update_product_traitlet()
    product_values = ExampleFactory.product.values

    assert sorted(factory_subclasses_str) == sorted(subclasses_str)
    assert sorted(factory_subclass_names) == sorted(subclass_names)
    assert sorted(product_values) == sorted(subclass_names)


def test_default():
    help_msg = ExampleFactory.class_get_help()
    assert ExampleFactory.default in help_msg
    obj = ExampleFactory().get_product()
    assert obj.__class__.__name__ == "ExampleComponent1"

    ExampleFactory.default = "ExampleComponent2"
    help_msg = ExampleFactory.class_get_help()
    assert ExampleFactory.default in help_msg
    obj = ExampleFactory().get_product()
    assert obj.__class__.__name__ == "ExampleComponent2"


def test_custom_product_help():
    help_msg = ExampleFactory.class_get_help()
    assert ExampleFactory.product_help in help_msg
    ExampleFactory.product_help += "2"
    help_msg = ExampleFactory.class_get_help()
    assert ExampleFactory.product_help in help_msg


def test_incorrect_factory_kwarg():
    with pytest.raises(TraitError):
        ExampleFactory(product='ExampleComponent2', value=111).get_product()


def test_false_product_name():
    with pytest.raises(KeyError):
        IncorrectFactory(product='ExampleComponent2').get_product()


def test_extra_config():
    config = Config()
    config['ExampleFactory'] = Config()
    config['ExampleFactory']['product'] = 'ExampleComponent2'
    config['ExampleComponent2'] = Config()
    config['ExampleComponent2']['value'] = 111
    config['ExampleComponent2']['extra'] = 4
    with pytest.warns(UserWarning):
        obj = ExampleFactory(config=config).get_product()
    assert obj.value == 111
    with pytest.raises(AttributeError):
        assert obj.extra == 4

    config['ExampleFactory'] = Config()
    config['ExampleFactory']['product'] = 'ExampleComponent4'
    config['ExampleComponent4'] = Config()
    config['ExampleComponent4']['value'] = 111
    config['ExampleComponent4']['extra'] = 4
    obj = ExampleFactory(config=config).get_product()
    assert obj.value == 111
    assert obj.extra == 4


def test_override_product():
    config = Config()
    config['ExampleFactory'] = Config()
    config['ExampleFactory']['product'] = 'ExampleComponent1'
    obj = ExampleFactory(
        config=config, product='ExampleComponent2'
    ).get_product()
    assert obj.__class__.__name__ == 'ExampleComponent2'


def test_trying_to_set_traitlets_via_factory():
    config = Config()
    config['ExampleFactory'] = Config()
    config['ExampleFactory']['product'] = 'ExampleComponent4'
    config['ExampleFactory']['value'] = 111
    config['ExampleFactory']['extra'] = 4
    with pytest.warns(UserWarning):
        obj = ExampleFactory(config=config).get_product()
    with pytest.raises(AssertionError):
        assert obj.value == 111
    with pytest.raises(AssertionError):
        assert obj.extra == 4


def test_component_definition_after_factory():
    """
    Test if a component defined after the factory definition will be
    obtainable by the factory, and is inside the help message
    """
    class ExampleComponent5(ExampleComponentParent):
        value = Int(1234445, help="").tag(config=True)

    obj = ExampleFactory(product='ExampleComponent5').get_product()
    assert isinstance(obj, ExampleComponent5)
    assert obj.value == 1234445
    assert "ExampleComponent5" in ExampleFactory.class_get_help()


def test_product_kwargs():
    obj = ExampleFactory(product='ExampleComponent2').get_product(value=5)
    assert isinstance(obj, ExampleComponent2)
    assert obj.value == 5


def test_product_kwargs_does_not_alter_config():
    config = Config()
    config['ExampleComponent2'] = Config()
    config['ExampleComponent2']['value'] = 111
    factory = ExampleFactory(config=config, product='ExampleComponent2')
    obj = factory.get_product(value=5)
    assert obj.value == 5
    assert factory.config['ExampleComponent2']['value'] == 111


def test_product_kwargs_override_config():
    config = Config()
    config['ExampleComponent2'] = Config()
    config['ExampleComponent2']['value'] = 111
    obj = ExampleFactory(
        config=config, product='ExampleComponent2'
    ).get_product(value=5)
    assert isinstance(obj, ExampleComponent2)
    assert obj.value == 5


def test_product_kwargs_override_config_parent():
    config = Config()
    config['ExampleComponentParent'] = Config()
    config['ExampleComponentParent']['value'] = 111
    obj = ExampleFactory(
        config=config, product='ExampleComponent2'
    ).get_product(value=5)
    assert isinstance(obj, ExampleComponent2)
    assert obj.value == 5


def test_product_kwargs_override_config_sibling():
    config = Config()
    config['ExampleComponent1'] = Config()
    config['ExampleComponent1']['value'] = 111
    obj = ExampleFactory(
        config=config, product='ExampleComponent2'
    ).get_product(value=5)
    assert isinstance(obj, ExampleComponent2)
    assert obj.value == 5


def test_product_kwargs_unrecognised():
    with pytest.warns(UserWarning):
        obj = ExampleFactory(product='ExampleComponent2').get_product(extra=5)
    assert isinstance(obj, ExampleComponent2)
    assert not hasattr(obj, 'extra')


def test_deprecated_behaviour():
    config = Config()
    config['ExampleFactory'] = Config()
    config['ExampleFactory']['product'] = 'ExampleComponent4'
    config['ExampleFactory']['value'] = 111
    config['ExampleFactory']['extra'] = 4
    with pytest.warns(DeprecationWarning):
        obj = ExampleFactory.produce(config=config)
    assert obj.value == 111
    assert obj.extra == 4

    with pytest.warns(DeprecationWarning):
        obj = ExampleFactory.produce(config=config, value=10000)
    assert obj.value == 10000
