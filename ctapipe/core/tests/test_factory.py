from ctapipe.core.factory import Factory
from ctapipe.core.component import Component
from traitlets import Int, TraitError
import pytest
from traitlets.config.loader import Config


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


class IncorrectExampleFactory(Factory):
    base = ExampleComponentParent
    default = 'ExampleComponent1'

    def _get_product_name(self):
        return "NonExistantClass"


def test_factory():
    obj = ExampleFactory.produce(
        config=None, tool=None,
        product='ExampleComponent2',
        value=111
    )
    assert(obj.__class__.__name__ == 'ExampleComponent2')
    assert(obj.value == 111)


def test_factory_subclass_detection():
    subclasses = [
        ExampleComponent1,
        ExampleComponent2,
        ExampleComponent3,
        ExampleComponent4
    ]
    subclass_names = [
        "ExampleComponent1",
        "ExampleComponent2",
        "ExampleComponent3",
        "ExampleComponent4"
    ]
    factory_subclasses_str = [str(i) for i in ExampleFactory.subclasses]
    subclasses_str = [str(i) for i in subclasses]
    assert sorted(factory_subclasses_str) == sorted(subclasses_str)
    assert sorted(ExampleFactory.subclass_names) == sorted(subclass_names)


def test_factory_automatic_traits():
    traits = sorted(list(ExampleFactory.class_own_traits().keys()))
    assert traits == sorted(['extra', 'product', 'value'])


def test_factory_traits_compatible_help():
    msg = [
        "Compatible Components:",
        "ExampleComponent1",
        "ExampleComponent2",
        "ExampleComponent3",
        "ExampleComponent4"
        ]
    for m in msg:
        assert m in ExampleFactory.class_own_traits()['value'].help


def test_factory_produce():
    obj = ExampleFactory.produce(config=None, tool=None,
                                 product='ExampleComponent2',
                                 value=111)
    assert (obj.__class__.__name__ == 'ExampleComponent2')
    assert (obj.value == 111)


def test_false_product_name():
    with pytest.raises(KeyError):
        IncorrectExampleFactory.produce(
            config=None, tool=None,
            product='ExampleComponent2',
            value=111
        )


def test_expected_args():
    kwargs = dict(
        product='ExampleComponent2',
        value=111,
        extra=4,
        nonexistant=5
    )
    with pytest.raises(TraitError):
        obj = ExampleFactory.produce(config=None, tool=None, **kwargs)

    kwargs.pop('nonexistant')
    obj = ExampleFactory.produce(config=None, tool=None, **kwargs)

    with pytest.raises(AttributeError):
        assert obj.extra == 4
    with pytest.raises(AttributeError):
        assert obj.nonexistant == 5

    kwargs['product'] = 'ExampleComponent3'
    obj = ExampleFactory.produce(config=None, tool=None, **kwargs)
    assert obj.extra == 4
    with pytest.raises(AttributeError):
        assert obj.nonexistant == 5

    kwargs['product'] = 'ExampleComponent4'
    obj = ExampleFactory.produce(config=None, tool=None, **kwargs)
    assert obj.extra == 4
    with pytest.raises(AttributeError):
        assert obj.nonexistant == 5


def test_expected_config():
    config = Config()
    config['ExampleFactory'] = Config()
    config['ExampleFactory']['product'] = 'ExampleComponent2'
    config['ExampleFactory']['value'] = 111
    config['ExampleFactory']['extra'] = 4
    obj = ExampleFactory.produce(config=config, tool=None)
    assert obj.value == 111
    with pytest.raises(AttributeError):
        assert obj.extra == 4

    config = Config()
    config['ExampleFactory'] = Config()
    config['ExampleFactory']['product'] = 'ExampleComponent2'
    config['ExampleComponent2'] = Config()
    config['ExampleComponent2']['value'] = 111
    config['ExampleComponent2']['extra'] = 4
    obj = ExampleFactory.produce(config=config, tool=None)
    assert obj.value == 111
    with pytest.raises(AttributeError):
        assert obj.extra == 4

    config = Config()
    config['ExampleFactory'] = Config()
    config['ExampleFactory']['product'] = 'ExampleComponent4'
    config['ExampleFactory']['value'] = 111
    config['ExampleFactory']['extra'] = 4
    obj = ExampleFactory.produce(config=config, tool=None)
    assert obj.value == 111
    assert obj.extra == 4

    config['ExampleFactory'] = Config()
    config['ExampleFactory']['product'] = 'ExampleComponent4'
    config['ExampleComponent4'] = Config()
    config['ExampleComponent4']['value'] = 111
    config['ExampleComponent4']['extra'] = 4
    obj = ExampleFactory.produce(config=config, tool=None)
    assert obj.value == 111
    assert obj.extra == 4
