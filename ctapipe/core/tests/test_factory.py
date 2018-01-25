from ctapipe.core.factory import Factory
from ctapipe.core.component import Component
from traitlets import Unicode, Int
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
    description = "Test Factory class"

    subclasses = Factory.child_subclasses(ExampleComponentParent)
    subclass_names = [c.__name__ for c in subclasses]

    discriminator = Unicode('ExampleComponent1',
                            help='Product to obtain: {}'
                            .format(subclass_names)).tag(config=True)

    # Product classes traits
    value = Int(555, help="").tag(config=True)

    def get_product_name(self):
        return self.discriminator


class IncorrectExampleFactory(ExampleFactory):
    def get_product_name(self):
        return "NonExistantClass"


def test_factory():
    obj = ExampleFactory.produce(
        config=None, tool=None,
        discriminator='ExampleComponent2',
        value=111
    )
    assert(obj.__class__.__name__ == 'ExampleComponent2')
    assert(obj.value == 111)


def test_factory_produce():
    obj = ExampleFactory.produce(config=None, tool=None,
                                 discriminator='ExampleComponent2',
                                 value=111)
    assert (obj.__class__.__name__ == 'ExampleComponent2')
    assert (obj.value == 111)


def test_false_product_name():
    with pytest.raises(KeyError):
        obj = IncorrectExampleFactory.produce(
            config=None, tool=None,
            discriminator='ExampleComponent2',
            value=111
        )


def test_expected_args():
    kwargs = dict(
        discriminator='ExampleComponent2',
        value=111,
        extra=4,
        nonexistant=5
    )
    obj = ExampleFactory.produce(config=None, tool=None, **kwargs)
    with pytest.raises(AttributeError):
        assert obj.extra == 4
    with pytest.raises(AttributeError):
        assert obj.nonexistant == 5

    kwargs['discriminator'] = 'ExampleComponent3'
    obj = ExampleFactory.produce(config=None, tool=None, **kwargs)
    assert obj.extra == 4
    with pytest.raises(AttributeError):
        assert obj.nonexistant == 5

    kwargs['discriminator'] = 'ExampleComponent4'
    obj = ExampleFactory.produce(config=None, tool=None, **kwargs)
    assert obj.extra == 4
    with pytest.raises(AttributeError):
        assert obj.nonexistant == 5


def test_expected_config():
    config = Config()
    config['ExampleFactory'] = Config()
    config['ExampleFactory']['discriminator'] = 'ExampleComponent2'
    config['ExampleFactory']['value'] = 111
    config['ExampleFactory']['extra'] = 4
    obj = ExampleFactory.produce(config=config, tool=None)
    assert obj.value == 111
    with pytest.raises(AttributeError):
        assert obj.extra == 4

    config = Config()
    config['ExampleFactory'] = Config()
    config['ExampleFactory']['discriminator'] = 'ExampleComponent2'
    config['ExampleComponent2'] = Config()
    config['ExampleComponent2']['value'] = 111
    config['ExampleComponent2']['extra'] = 4
    obj = ExampleFactory.produce(config=config, tool=None)
    assert obj.value == 111
    with pytest.raises(AttributeError):
        assert obj.extra == 4

    config = Config()
    config['ExampleFactory'] = Config()
    config['ExampleFactory']['discriminator'] = 'ExampleComponent4'
    config['ExampleFactory']['value'] = 111
    config['ExampleFactory']['extra'] = 4
    obj = ExampleFactory.produce(config=config, tool=None)
    assert obj.value == 111
    assert obj.extra == 4

    config['ExampleFactory'] = Config()
    config['ExampleFactory']['discriminator'] = 'ExampleComponent4'
    config['ExampleComponent4'] = Config()
    config['ExampleComponent4']['value'] = 111
    config['ExampleComponent4']['extra'] = 4
    obj = ExampleFactory.produce(config=config, tool=None)
    assert obj.value == 111
    assert obj.extra == 4
