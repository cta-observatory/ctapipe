from copy import deepcopy

from ctapipe.core.component import Component
from abc import abstractmethod


class Factory(Component):
    # TODO: obtain traits automatically for the classes that can be produced
    """
    A base class for all class factories that exist in the `Tools`/`Components`
    frameworks.

    Either maunally set "subclasses" as a list of the classes that you wish
    to be inside the factory, or set it to
    Factory.child_subclasses(ParentClass). Using that static method will
    obtain all the lowest-level child classes from the parent class.

    You must manually specify the discriminator trait (which is what you
    set at run-time to choose the product you wish to obtain from the factory)
    and the traits of all the classes that are possible to obtain from the
    factory. Perhaps in the future these can be obtained automatically.

    You must also return the factory name in get_factory_name(), and the
    discriminator in get_product_name() in your custom Factory class.

    To then obtain the product class from the factory, use 'get_class()",
    which can be used to then initialise the class. The correct traits
    from the correspoding product are set automatically from the factory.

    .. code:: python

        from ctapipe.core import Factory
        from traitlets import (Integer, Float, List, Dict, Unicode)

        class MyFactory(Factory):
            name = "myfactory"
            description = "do some things and stuff"

            subclasses = Factory.child_subclasses(ParentClass)
            subclass_names = [c.__name__ for c in subclasses]

            discriminator = Unicode('DefaultProduct',
                             help='Product to obtain: {}'
                             .format(subclass_names)).tag(config=True)

            # Product classes traits
            # Would be nice to have these automatically set...!
            product_trait1 = Int(7, help="").tag(config=True)
            product_trait2 = Int(7, help="").tag(config=True)

            def get_factory_name(self):
                return self.name

            def get_product_name(self):
                return self.discriminator
    """
    subclasses = None  # Set to all_subclasses(ParentClass) in factory
    subclass_names = None  # Set to [c.__name__ for c in subclasses] in factory

    def __init__(self, config, tool, **kwargs):
        """
        Base Factory class

        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool : ctapipe.core.Tool
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        kwargs

        """
        super().__init__(config=config, parent=tool, **kwargs)
        self.product = None

    @staticmethod
    def child_subclasses(cls):
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
        family = cls.__subclasses__() + [g for s in cls.__subclasses__()
                                         for g in Factory.child_subclasses(s)]
        children = [g for g in family if not g.__subclasses__()]
        return children

    @abstractmethod
    def get_product_name(self):
        """
        Abstract method to be implemented in child factory.
        Simply return the discriminator traitlet.
        """

    @abstractmethod
    def get_factory_name(self):
        """
        Abstract method to be implemented in child factory.
        Simply return the name of the factory.
        """

    def get_class(self):
        """
        Obtain the class constructor for the specified product name.

        Returns
        -------
        product : class

        """
        subclass_dict = dict(zip(self.subclass_names, self.subclasses))
        self.log.info("Obtaining {} from {}".format(self.get_product_name(),
                                                    self.get_factory_name()))
        try:
            product = subclass_dict[self.get_product_name()]
        except KeyError:
            self.log.exception('No product found with name "{}" for '
                               'factory.'.format(self.get_product_name()))
            raise

        # Copy factory traits to product
        c = self.__dict__['_trait_values']['config']
        c[product.name] = deepcopy(c[self.get_factory_name()])
        items = self.__dict__['_trait_values'].items()
        for key, values in items:
            if key != 'config' and key != 'parent':
                c[product.name][key] = values
        keys = list(c[product.name].keys())
        for key in keys:
            if key not in product.class_trait_names():
                del c[product.name][key]
        return product
