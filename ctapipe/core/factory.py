from copy import copy
from ctapipe.core.component import Component
from inspect import isabstract
from traitlets import CaselessStrEnum
import warnings


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
        for n, g in child_subclasses(s).items()
    ]
    children = {g.__name__: g for g in family if not isabstract(g)}

    return children


class FactoryMeta(type(Component), type):
    def __new__(mcs, name, bases, dct):
        """
        This metaclass is required to create a fresh `product` traitlet for
        each `Factory`. If the traitlet was instead defined in the `Factory`
        baseclass, then every `Factory` would share the same traitlet, and
        problems would occur.
        The test 'ctapipe/core/tests/test_factory.py::
        test_second_factory_product_different' checks for this.
        """
        # Setup class lookup
        base = dct['base']
        if base:
            dct['product'] = CaselessStrEnum(
                [],
                None,
                allow_none=True,
                help=''
            ).tag(config=True)
        return type.__new__(mcs, name, bases, dct)


class Factory(Component, metaclass=FactoryMeta):
    """
    A base class for all class factories that exist in the `Tools`/`Components`
    frameworks.

    To create a Factory, inherit this class and set `base` to the
    base-class of the Components.

    A `product` traitlet is automatically generated which allows selection of
    the returned `Component` class by name. The default `Component` can be set
    by the `default` attribute of the `Factory`. The help message that appears
    for the `product` traitlet in a `Tool` utilising the `Factory` can be set
    with the `product_help` attribute.

    To use a factory from within a `ctapipe.core.tool.Tool`:

    >>> cls = Factory(config=self.config, tool=self).produce()

    Arguments can be passed to the produced `Component` via the arguments to
    `produce`.

    Attributes
    ----------
    base : type
        The base-class for the different classes that could be returned from
        the factory
    default : str
        The default product to return. If None, then there is no default.
    product_help : None or str
        If set, then this text will be displayed for the help message for the
        product traitlet.
    product : traitlets.CaselessStrEnum
        The traitlet allowing the manual specification of which class is
        returned from the factory.
    """
    base = None
    default = None
    product_help = 'Product class to obtain from the Factory.'
    product = None  # Defined by metaclass

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
        cls.product.values = list(child_subclasses(cls.base).keys())
        cls.product.default_value = cls.default
        cls.product.help = cls.product_help

    @classmethod
    def class_get_help(cls, inst=None):
        """
        Update values before obtaining help message to make sure it contains
        Components included since Factory definition
        """
        cls.update_product_traitlet()
        return super().class_get_help(inst)

    def _clean_kwargs_for_product(self, kwargs_dict):
        """
        Remove and warn about kwargs that would throw an error for this
        particular product.

        There may be a usecase for passing arguments to the produce method of
        the `Factory`, but the arguments are only valid for some of the
        possible products of the Factory. This may be because some Components
        have different traitlets. This function therefore removes arguments
        that are not valid for the product Component returned this time from
        the Factory.

        Parameters
        ----------
        kwargs_dict : dict
            The full kwargs dictionary

        Returns
        -------
        kwargs_dict : dict
            The kwargs dictionary with the non-compatible arguments removed
        """
        product = self._product
        product_traits = product.class_trait_names()
        product_args = list(product.__init__.__code__.co_varnames)
        # config = copy(self.__dict__['_trait_values']['config'])
        # parent = copy(self.__dict__['_trait_values']['parent'])

        # Copy valid arguments to kwargs
        kwargs_copy = copy(kwargs_dict)
        for key in list(kwargs_copy.keys()):
            if key not in product_traits + product_args:
                del kwargs_copy[key]
                msg = ("Traitlet ({}) does not exist for {}"
                       .format(key, product.__name__))
                self.log.warning(msg)
                warnings.warn(msg, stacklevel=9)
        return kwargs_copy

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
        kwargs = self._clean_kwargs_for_product(kwargs)
        instance = self._product(self.config, self.parent, **kwargs)
        return instance
