from ctapipe.core.component import Component
from abc import abstractmethod


class Factory(Component):
    """
    A base class for all class factories that exist in the `Tools`/`Components`
    frameworks.

    It operates on the assumption that all possible class products of the
    factory are a base child of a singular parent class, and have no further
    children of themselves.

    When the factory class is specified in the `factories` Dict inside a `Tool`
    the factory discriminator trailet is evaluated, and the resultant product
    class is obtained (using `init_product()`) and added to the `classes` List.
    All the traitlets of the class is automatically added to the `aliases`
    Dict. This allows dynamic definition of command-line arguments depending
    on the factory discriminator traitlet.

    To then obtain an instance of the product class, use `get_product()`, an
    abstract method that should be defined with the relavant arguments for
    the class.

    .. code:: python

    from ctapipe.core import Factory
    from traitlets import (Integer, Float, List, Dict, Unicode)

    class MyFactory(Factory):
        name = "myfactory"
        description = "do some things and stuff"

        subclasses = Factory.all_subclasses(ParentClass)
        subclass_names = [c.__name__ for c in subclasses]

        discriminator = Unicode('DefaultProduct',
                                 help='Product to obtain: {}'
                                 .format(subclass_names)).tag(config=True)

        def get_product_name(self):
            return self.discriminator

        def get_product(self, product_args=None, parent=None,
                        config=None, **kwargs):
            if not self.product:
                self.init_product()
            product = self.product
            object_instance = product(product_args, parent=parent,
                                      config=config, **kwargs)
            return object_instance


    """
    subclasses = None  # Set to all_subclasses(ParentClass) in factory
    subclass_names = None  # Set to [c.__name__ for c in subclasses] in factory

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.product = None

    @staticmethod
    def all_subclasses(cls):
        """
        Return all base subclasses of a parent class. Finds the bottom level
        subclasses that have no further children.
        Parameters
        ----------
        cls : class
            high level class object that contains the desired children classes

        Returns
        -------
        list
            list of bottom level subclasses

        """
        return cls.__subclasses__() + [g for s in cls.__subclasses__()
                                       for g in Factory.all_subclasses(s)]

    @abstractmethod
    def get_product_name(self):
        """
        Abstract method to be implemented in child factory.
        Simply return the discriminator traitlet.
        """

    def init_product(self, product_name=None):
        if not product_name:
            product_name = self.get_product_name()
        for subclass in self.subclasses:
            if subclass.__name__ == product_name:
                self.product = subclass
                return subclass
        raise KeyError('No subclass exists with name: '
                       '{}'.format(self.get_product_name))

    @abstractmethod
    def get_product(self):
        """
        Abstract method to be implemented in child factory.
        If self.product has not been set, then call self.init_product. Then
        return a instance of product with the correct arguments passed to it
        (especially config=config).

        All implementations of this function must have all arguments set with a
        default so LSV is not violated:
        https://en.wikipedia.org/wiki/Liskov_substitution_principle
        """
        pass
