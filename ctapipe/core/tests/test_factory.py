from ctapipe.core.factory import Factory
from ctapipe.core.component import Component
from traitlets import Unicode, Int


class ExampleComponentParent(Component):
    value = Int(123, help="").tag(config=True)


class ExampleComponent1(ExampleComponentParent):
    value = Int(123111, help="").tag(config=True)


class ExampleComponent2(ExampleComponentParent):
    value = Int(123222, help="").tag(config=True)


class ExampleFactory(Factory):
    description = "Test Factory class"

    subclasses = Factory.child_subclasses(ExampleComponentParent)
    subclass_names = [c.__name__ for c in subclasses]

    discriminator = Unicode('ExampleComponent1',
                            help='Product to obtain: {}'
                            .format(subclass_names)).tag(config=True)

    # Product classes traits
    value = Int(555, help="").tag(config=True)

    def get_factory_name(self):
        return self.__class__.__name__

    def get_product_name(self):
        return self.discriminator


def test_factory():
    factory = ExampleFactory(config=None, tool=None)
    factory.discriminator = 'ExampleComponent2'
    factory.value = 111
    cls = factory.get_class()
    obj = cls(config=factory.config, parent=None)
    assert(obj.__class__.__name__ == 'ExampleComponent2')
    assert(obj.value == 111)


def test_factory_produce():
    obj = ExampleFactory.produce(config=None, tool=None,
                                 discriminator='ExampleComponent2',
                                 value=111)
    assert (obj.__class__.__name__ == 'ExampleComponent2')
    assert (obj.value == 111)
