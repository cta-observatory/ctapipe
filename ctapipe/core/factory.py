from copy import deepcopy

from ctapipe.core.component import Component
from abc import abstractmethod
from inspect import isabstract
from traitlets.config.loader import Config


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

    To then obtain the product class from the factory, use 'produce()".
    The correct traits from the correspoding product are set automatically
    from the factory.

    To a factory from within a `ctapipe.core.tool.Tool`:

    >>> cls = FactoryChild.produce(config=self.config, tool=self)

    .. code:: python

        from ctapipe.core import Factory
        from traitlets import (Integer, Float, List, Dict, Unicode)

        class MyFactory(Factory):
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
                return self.__class__.__name__

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
        self.kwargs = deepcopy(kwargs)

    @staticmethod
    def child_subclasses(cls):
        """
        Return all non-abstract subclasses of a parent class.

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
        children = [g for g in family if not isabstract(g)]

        return children

    @abstractmethod
    def get_product_name(self):
        """
        Abstract method to be implemented in child factory.

        Method to obtain the correct name for the product.
        """

    @property
    def _product(self):
        """
        Obtain the class constructor for the specified product name.

        Returns
        -------
        product : class

        """
        subclass_dict = dict(zip(self.subclass_names, self.subclasses))
        self.log.info("Obtaining {} from {}".format(self.get_product_name(),
                                                    self.__class__.__name__))
        try:
            product = subclass_dict[self.get_product_name()]
            return product
        except KeyError:
            self.log.exception('No product found with name "{}" for '
                               'factory.'.format(self.get_product_name()))
            raise

    @property
    def _instance(self):
        """
        Obtain an instance of the product class with the config and arguments
        correctly copied from the Factory to the product.

        Returns
        -------
        instance
            Instance of the product class that is the purpose of the factory
            to produce.
        """
        product = self._product
        product_traits = product.class_trait_names()
        product_args = list(product.__init__.__code__.co_varnames)
        config = deepcopy(self.__dict__['_trait_values']['config'])
        parent = deepcopy(self.__dict__['_trait_values']['parent'])

        if config[self.__class__.__name__]:
            # If Product config does not exist, create new Config instance
            # Note: `config[product.__name__]` requires Config, not dict
            if not config[product.__name__]:
                config[product.__name__] = Config()

            # Copy Factory config to Product config
            for key, value in config[self.__class__.__name__].items():
                if key in product_traits:
                    config[product.__name__][key] = value

        # Copy valid arguments to kwargs
        kwargs = deepcopy(self.kwargs)
        for key in list(kwargs.keys()):
            if key not in product_traits + product_args:
                del kwargs[key]

        instance = product(config, parent, **kwargs)
        return instance

    @classmethod
    def produce(cls, config, tool, **kwargs):
        """
        Produce an instance of the product class

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

        Returns
        -------
        instance
            Instance of the product class that is the purpose of the factory
            to produce.

        """
        factory = cls(config=config, tool=tool, **kwargs)
        instance = factory._instance
        return instance
