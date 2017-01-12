from ctapipe.core.factory import Factory
from ctapipe.core.component import Component
from traitlets import Unicode, Int


class TestComponentParent(Component):
    name = 'TestComponentParent'
    value = Int(123, help="").tag(config=True)


class TestComponent1(TestComponentParent):
    name = 'TestComponent1'
    value = Int(123111, help="").tag(config=True)


class TestComponent2(TestComponentParent):
    name = 'TestComponent2'
    value = Int(123222, help="").tag(config=True)


class TestFactory(Factory):
    name = 'TestFactory'
    description = "Test Factory class"

    subclasses = Factory.child_subclasses(TestComponentParent)
    subclass_names = [c.__name__ for c in subclasses]

    discriminator = Unicode('TestComponent1',
                            help='Product to obtain: {}'
                            .format(subclass_names)).tag(config=True)

    # Product classes traits
    value = Int(555, help="").tag(config=True)

    def get_factory_name(self):
        return self.name

    def get_product_name(self):
        return self.discriminator


def test_factory():
    factory = TestFactory(config=None, tool=None)
    factory.discriminator = 'TestComponent2'
    factory.value = 111
    cls = factory.get_class()
    obj = cls(config=factory.config, parent=None)
    assert(obj.name == 'TestComponent2')
    assert(obj.value == 111)
