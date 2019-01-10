from copy import copy, deepcopy
from ctapipe.core.component import Component
from inspect import isabstract
from traitlets.config.loader import Config
from traitlets import CaselessStrEnum


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
    children : dict
        list of non-abstract subclasses

    """
    family = base.__subclasses__() + [
        g for s in base.__subclasses__()
        for g in child_subclasses(s)
    ]
    children = {g.__name__: g for g in family if not isabstract(g)}

    return children


class Factory(Component):
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
    default = None
    custom_product_help = 'Product class to obtain from the Factory.'
    # TODO: Rename custom_product_help to product_help

    product = CaselessStrEnum(
        [],
        default,
        allow_none=True,
        help=custom_product_help
    ).tag(config=True)

    def __new__(cls, *args, **kwargs):
        """
        Setup product traitlet
        Also ensures the values of the product traitlet contain Components
        defined since Factory definition
        """
        cls.update_product_traitlet()
        cls = super().__new__(cls, *args, **kwargs)
        return cls

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

    @classmethod
    def update_product_traitlet(cls):
        """
        Update the values for the product trailet so they match the properties
        of the Factory and the loaded Components
        """
        cls.product.values = child_subclasses(cls.base).keys()
        cls.product.default_value = cls.default
        cls.product.help = cls.custom_product_help

    @classmethod
    def class_get_help(cls, inst=None):
        """
        Update values before obtaining help message to make sure it contains
        Components included since Factory definition
        """
        cls.update_product_traitlet()
        return super().class_get_help(inst)

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
        subclass_dict = child_subclasses(self.base)
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

    def produce(self, **kwargs):
        """
        Produce an instance of the product class from the Factory

        Parameters
        ----------
        kwargs
            Named arguments to pass to product

        Returns
        -------
        instance
            Instance of the product class that is the purpose of the factory
            to produce.

        """
        product = self._product
        product_traits = product.class_trait_names()
        product_args = list(product.__init__.__code__.co_varnames)
        # config = copy(self.__dict__['_trait_values']['config'])
        # parent = copy(self.__dict__['_trait_values']['parent'])

        # Copy valid arguments to kwargs
        for key in list(kwargs.keys()):
            if key not in product_traits + product_args:
                del kwargs[key]
                self.log.warning("Traitlet ({}) does not be exist for {}"
                                 .format(key, product.__name__))

        instance = product(self.config, self.parent, **kwargs)
        return instance
