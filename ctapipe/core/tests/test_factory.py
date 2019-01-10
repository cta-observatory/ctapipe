from ctapipe.core.factory import Factory, child_subclasses
from ctapipe.core.component import Component
from traitlets import Int, TraitError
import pytest
from traitlets.config.loader import Config
import warnings


class ExampleComponentParent(Component):
    value = Int(123, help="").tag(config=True)

    def __init__(self, config, parent, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)


class ExampleComponent1(ExampleComponentParent):
    value = Int(123111, help="").tag(config=True)


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
    custom_product_help = "Product for testing"


class IncorrectExampleFactory(Factory):
    base = ExampleComponentParent
    default = 'ExampleComponent1'

    def _get_product_name(self):
        return "NonExistantClass"


def test_factory():
    obj = ExampleFactory(product='ExampleComponent2').produce(value=111)
    assert(obj.__class__.__name__ == 'ExampleComponent2')
    assert(obj.value == 111)


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
    obj = ExampleFactory().produce()
    assert obj.__class__.__name__ == "ExampleComponent1"

    ExampleFactory.default = "ExampleComponent2"
    help_msg = ExampleFactory.class_get_help()
    assert ExampleFactory.default in help_msg
    obj = ExampleFactory().produce()
    assert obj.__class__.__name__ == "ExampleComponent2"


def test_custom_product_help():
    help_msg = ExampleFactory.class_get_help()
    assert ExampleFactory.custom_product_help in help_msg
    ExampleFactory.custom_product_help += "2"
    help_msg = ExampleFactory.class_get_help()
    assert ExampleFactory.custom_product_help in help_msg


def test_factory_produce():
    with pytest.raises(TraitError): # TODO: Change to `with pytest.warns(UserWarning):`, but doesn't work?
        ExampleFactory().produce(product='ExampleComponent2',value=111)

    with pytest.raises(TraitError):
        ExampleFactory(product='ExampleComponent2', value=111).produce()

    obj = ExampleFactory(product='ExampleComponent2').produce(value=111)
    assert (obj.__class__.__name__ == 'ExampleComponent2')
    assert (obj.value == 111)


def test_false_product_name():
    with pytest.raises(KeyError):
        IncorrectExampleFactory(product='ExampleComponent2').produce(value=111)


def test_expected_args():
    fkwargs = dict(
        product='ExampleComponent2',
    )
    pkwargs = dict(
        value=111,
        extra=4,
        nonexistant=5
    )
    with pytest.raises(TraitError):
        obj = ExampleFactory(**fkwargs).produce(**pkwargs)

    pkwargs.pop('nonexistant')
    obj = ExampleFactory(**fkwargs).produce(**pkwargs)

    with pytest.raises(AttributeError):
        assert obj.extra == 4
    with pytest.raises(AttributeError):
        assert obj.nonexistant == 5

    fkwargs['product'] = 'ExampleComponent3'
    obj = ExampleFactory(**fkwargs).produce(**pkwargs)
    assert obj.extra == 4
    with pytest.raises(AttributeError):
        assert obj.nonexistant == 5

    fkwargs['product'] = 'ExampleComponent4'
    obj = ExampleFactory(**fkwargs).produce(**pkwargs)
    assert obj.extra == 4
    with pytest.raises(AttributeError):
        assert obj.nonexistant == 5


def test_expected_config():
    config = Config()
    config['ExampleFactory'] = Config()
    config['ExampleFactory']['product'] = 'ExampleComponent2'
    config['ExampleFactory']['value'] = 111
    config['ExampleFactory']['extra'] = 4
    obj = ExampleFactory(config=config).produce()
    with pytest.raises(AssertionError):
        # Cannot set Component value via Factory
        assert obj.value == 111
    with pytest.raises(AttributeError):
        assert obj.extra == 4

    config = Config()
    config['ExampleFactory'] = Config()
    config['ExampleFactory']['product'] = 'ExampleComponent2'
    config['ExampleComponent2'] = Config()
    config['ExampleComponent2']['value'] = 111
    config['ExampleComponent2']['extra'] = 4
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=UserWarning)
        obj = ExampleFactory(config=config).produce()
    assert obj.value == 111
    with pytest.raises(AttributeError):
        assert obj.extra == 4

    config = Config()
    config['ExampleFactory'] = Config()
    config['ExampleFactory']['product'] = 'ExampleComponent4'
    config['ExampleFactory']['value'] = 111
    config['ExampleFactory']['extra'] = 4
    obj = ExampleFactory(config=config).produce()
    with pytest.raises(AssertionError):
        # Cannot set Component value via Factory
        assert obj.value == 111
    with pytest.raises(AssertionError):
        # Cannot set Component value via Factory
        assert obj.extra == 4

    config['ExampleFactory'] = Config()
    config['ExampleFactory']['product'] = 'ExampleComponent4'
    config['ExampleComponent4'] = Config()
    config['ExampleComponent4']['value'] = 111
    config['ExampleComponent4']['extra'] = 4
    obj = ExampleFactory(config=config).produce()
    assert obj.value == 111
    assert obj.extra == 4


def test_component_definition_after_factory():
    """
    Test if a component defined after the factory definition will be
    obtainable by the factory
    """
    class ExampleComponent5(ExampleComponentParent):
        value = Int(1234445, help="").tag(config=True)

    obj = ExampleFactory(product='ExampleComponent5').produce()
    assert (obj.__class__.__name__ == 'ExampleComponent5')
    assert (obj.value == 1234445)


def test_component_definition_after_factory_help_message():
    """
    Test if a component defined after the factory definition will be
    inside the help message
    """
    class ExampleComponent6(ExampleComponentParent):
        value = Int(1234446, help="").tag(config=True)

    assert "ExampleComponent6" in ExampleFactory.class_get_help()
