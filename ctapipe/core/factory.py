from copy import copy, deepcopy
from ctapipe.core.component import Component
from inspect import isabstract
from traitlets.config.loader import Config
from traitlets import CaselessStrEnum


class FactoryMeta(type(Component), type):
    def __new__(mcs, name, bases, dct):

        # Setup class lookup
        base = dct['base']
        dct['subclasses'] = None
        dct['subclass_names'] = None
        if base:
            if not isinstance(base, type) or not issubclass(base, Component):
                raise AttributeError("Factory.base must be set to a Component")

            dct['subclasses'] = mcs.child_subclasses(base)
            dct['subclass_names'] = [c.__name__ for c in dct['subclasses']]

            default = None if 'default' not in dct else dct['default']
            help_msg = 'Product class to obtain from the Factory.'
            if 'custom_product_help' in dct and dct['custom_product_help']:
                help_msg = dct['custom_product_help']
            dct['product'] = CaselessStrEnum(
                dct['subclass_names'],
                default,
                allow_none=True,
                help=help_msg
            ).tag(config=True)

            # Gather a record of which traits are valid for which subclasses
            # and copy subclass traits
            traits = dict()
            record = dict()
            for sub in dct['subclasses']:
                for key, trait in sub.class_traits().items():
                    if key in ['config', 'parent']:
                        continue
                    record.setdefault(key, []).append(sub.__name__)
                    if key not in traits:
                        traits[key] = deepcopy(trait)

            # Add subclass traits to Factory and include a list of valid
            # sublasses for the trait help message
            for key, trait in traits.items():
                trait.help += "\n\nCompatible Components: " + str(record[key])
                dct[key] = trait

        return type.__new__(mcs, name, bases, dct)

    @staticmethod
    def child_subclasses(base):
        """
        Return all non-abstract subclasses of a base class.

        Parameters
        ----------
        base : class
            high level class object that is inherited by the
            desired subclasses

        Returns
        -------
        children : list
            list of non-abstract subclasses

        """
        family = base.__subclasses__() + [
            g for s in base.__subclasses__()
            for g in FactoryMeta.child_subclasses(s)
        ]
        children = [g for g in family if not isabstract(g)]

        return children


class Factory(Component, metaclass=FactoryMeta):
    """
    A base class for all class factories that exist in the `Tools`/`Components`
    frameworks.

    To create a Factory, inherit this class and set `base` to the
    base-class of the Factory.

    The traits of the sub-classes are automatically added to the Factory
    (and are included in the help message).

    The traits and kwargsthat correctly correspond to the product are passed
    onto the product.

    To use a factory from within a `ctapipe.core.tool.Tool`:

    >>> cls = Factory.produce(config=self.config, tool=self)

    Attributes
    ----------
    base : type
        The base-class for the different classes that could be returned from
        the factory
    product : traitlets.CaselessStrEnum
        The traitlet allowing the manual specification of which class is
        returned from the factory. This is created in `FactoryMeta`.
    subclasses : list
        A list of the subclasses for the base-class specified in `base`
    subclass_names : list
        The names of the classes in `subclasses`
    default : str
        The default product to return. If None, then there is no default.
    custom_product_help : None or str
        If set, then this text will be displayed for the help message for the
        product traitlet.
    """
    base = None
    product = None  # Instanced as a traitlet by FactoryMeta
    subclasses = None  # Filled by FactoryMeta
    subclass_names = None  # Filled by FactoryMeta
    default = None
    custom_product_help = None

    def __init__(self, config=None, tool=None, **kwargs):
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
        if not self.base:
            raise AttributeError("Factory.base must be set to a Component")

        super().__init__(config=config, parent=tool, **kwargs)
        self.kwargs = copy(kwargs)

    def _get_product_name(self):
        """
        Method to obtain the correct name for the product.

        Returns
        -------
        str
            The name of the product to return from the Factory.
        """
        if self.product:
            return self.product
        else:
            raise AttributeError("The user has not specified a product for {}"
                                 .format(self.__class__.__name__))

    @property
    def _product(self):
        """
        Obtain the class constructor for the specified product name.

        Returns
        -------
        product : class

        """
        subclass_dict = dict(zip(self.subclass_names, self.subclasses))
        product_name = self._get_product_name()
        self.log.info("Obtaining {} from {}".format(product_name,
                                                    self.__class__.__name__))
        try:
            product = subclass_dict[product_name]
            return product
        except KeyError:
            self.log.exception('No product found with name "{}" for '
                               'factory.'.format(product_name))
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
        config = copy(self.__dict__['_trait_values']['config'])
        parent = copy(self.__dict__['_trait_values']['parent'])

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
        kwargs = copy(self.kwargs)
        for key in list(kwargs.keys()):
            if key not in product_traits + product_args:
                del kwargs[key]

        instance = product(config, parent, **kwargs)
        return instance

    @classmethod
    def produce(cls, config=None, tool=None, **kwargs):
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
