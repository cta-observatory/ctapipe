from copy import copy
from ctapipe.core.component import Component
from inspect import isabstract
from traitlets import CaselessStrEnum
import warnings
from traitlets.config.loader import Config


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
    non_abstract : dict
        dict of all non-abstract subclasses

    """
    subclasses = base.__subclasses__() + [
        g for s in base.__subclasses__()
        for n, g in child_subclasses(s).items()
    ]
    non_abstract = {g.__name__: g for g in subclasses if not isabstract(g)}

    return non_abstract


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

    Below is an example of a `Factory` used witin a `ctapipe.core.tool.Tool`:

    >>> from traitlets import Dict, List, Int
    >>> from ctapipe.core import Tool, Component, Factory
    >>>
    >>>
    >>> class ExampleComponentParent(Component):
    >>>     value = Int(123, help="").tag(config=True)
    >>>
    >>>     def __init__(self, config, tool, **kwargs):
    >>>         super().__init__(config=config, parent=tool, **kwargs)
    >>>
    >>>
    >>> class ExampleComponent1(ExampleComponentParent):
    >>>     value = Int(123111, help="").tag(config=True)
    >>>
    >>>
    >>> class ExampleComponent2(ExampleComponentParent):
    >>>     value = Int(123222, help="").tag(config=True)
    >>>
    >>>
    >>> class ExampleFactory(Factory):
    >>>     base = ExampleComponentParent
    >>>     default = 'ExampleComponent1'
    >>>
    >>>
    >>> class ExampleTool(Tool):
    >>>     name = "ExampleTool"
    >>>
    >>>     aliases = Dict(dict(
    >>>         product='ExampleFactory.product',
    >>>     ))
    >>>
    >>>     classes = List([
    >>>         ExampleFactory,
    >>>     ])
    >>>
    >>>     def setup(self):
    >>>         kwargs = dict(config=self.config, tool=self)
    >>>
    >>>         example = ExampleFactory(**kwargs).get_product()
    >>>         print(example.__class__.__name__)
    >>>
    >>>     def start(self):
    >>>         pass
    >>>
    >>>     def finish(self):
    >>>         pass

    Arguments can be passed to the produced `Component` via the arguments to
    `produce`.

    Parameters
    ----------
    config : traitlets.loader.Config
        Configuration specified by config file or cmdline arguments.
        This argument is typically only used from within a `ctapipe.core.Tool`.
        Used to set traitlet values.
        Leave as None if no configuration to pass.
    tool : ctapipe.core.Tool or None
        Tool executable that is calling this component.
        This argument is typically only used from within a `ctapipe.core.Tool`.
        Passes the correct logger to the component.
        Leave as None if no Tool to pass.
    kwargs
        Named arguments to pass to the Factory. These are not passed on
        to the product Component. Use the arguments to `produce` to pass
        arguments to a Component.

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
        Setup product traitlet before instancing of Factory.
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
            Named arguments to pass to the Factory. These are not passed on
            to the product Component. Use the arguments to `produce` to pass
            arguments to a Component.
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
    def class_get_trait_help(cls, trait, inst=None):
        """
        Override of Configurable.class_get_trait_help to update values before
        obtaining help message to make sure it contains Components included
        since Factory definition
        """
        if trait == cls.product:
            cls.update_product_traitlet()
        return super().class_get_trait_help(trait, inst)

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

    def _get_product_constructor(self, product_name):
        """
        Obtain the class constructor for the specified product name.

        Parameters
        ----------
        product_name : str
            Name of the subclass to obtain from the factory

        Returns
        -------
        product : class

        """
        subclass_dict = child_subclasses(self.base)
        self.log.info("Obtaining {} from {}".format(product_name,
                                                    self.__class__.__name__))
        try:
            product = subclass_dict[product_name]
            return product
        except KeyError:
            self.log.exception('No product found with name "{}" for '
                               'factory.'.format(product_name))
            raise

    def get_product(self):
        """
        Produce an instance of the product class from the Factory

        Returns
        -------
        product_instance
            Instance of the product class that is the purpose of the factory
            to produce.

        """
        product_name = self._get_product_name()
        product_constructor = self._get_product_constructor(product_name)
        product_instance = product_constructor(self.config, self.parent)
        return product_instance

    @classmethod
    def produce(cls, config=None, tool=None, **kwargs):
        """
        Deprecated method to produce the product of the Factory via the
        classmethod. Also contains the old behaviour of copying the traits
        specified for the Factory to the Component.

        This method therefore avoids a change in API being forced on users
        who have not updated their scripts since PR #917. Instead they are
        warned that they need to do so.

        Instead of this classmethod, one should switch to the new API
        using the get_product method after the Factory has been initialised.
        See the Factory docstring for an example, and PR #917 if you want
        more details.

        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            This argument is typically only used from within a
            `ctapipe.core.Tool`.
            Used to set traitlet values.
            Leave as None if no configuration to pass.
        tool : ctapipe.core.Tool or None
            Tool executable that is calling this component.
            This argument is typically only used from within a
            `ctapipe.core.Tool`.
            Passes the correct logger to the component.
            Leave as None if no Tool to pass.
        kwargs

        Returns
        -------
        instance
            Instance of the product class that is the purpose of the factory
            to produce.

        """
        msg = (
            "The produce classmethod of Factory has been deprecated. "
            "Instead of:\n"
            "\tcls = Factory.produce(config=self.config, tool=self, "
            "**product_kwargs)\n"
            "Please switch to:\n"
            "\tcls = Factory(config=self.config, tool=self, "
            "**factory_kwargs).get_product(**product_kwargs)\n"
            "See the doctring of Factory for an example, and "
            "https://github.com/cta-observatory/ctapipe/pull/917 "
            "for further details."
        )
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter

        # Remove kwargs not meant for Factory
        factory_traits = cls.class_trait_names()
        factory_args = list(cls.__init__.__code__.co_varnames)
        factory_kwargs = copy(kwargs)
        for key in list(factory_kwargs.keys()):
            if key not in factory_traits + factory_args:
                del factory_kwargs[key]

        # Obtain Component constructor
        factory = cls(config=config, tool=tool, **factory_kwargs)
        if hasattr(factory, 'input_url'):
            kwargs['input_url'] = factory.input_url
        try:
            product_name = factory._get_product_name()
            constructor = factory._get_product_constructor(product_name)
        except ValueError:
            if hasattr(cls, 'input_url'):
                msg = ("Passing the input_url to the initialisation of "
                       "EventSourceFactory, and then calling the deprecated "
                       "`produce` classmethod is not supported. Please switch "
                       "to the `get_product` method. See the "
                       "EventSourceFactory docstring for examples")
                raise SyntaxError(msg)
            raise

        # Copy traits accidently passed to Factory to Component
        if config and config[cls.__name__]:
            if not config[constructor.__name__]:
                config[constructor.__name__] = Config()
            for key, value in config[cls.__name__].items():
                if key in constructor.class_trait_names():
                    config[constructor.__name__][key] = value

        # Remove kwargs not meant for product
        product_traits = constructor.class_trait_names()
        product_args = list(constructor.__init__.__code__.co_varnames)
        product_kwargs = copy(kwargs)
        for key in list(product_kwargs.keys()):
            if key not in product_traits + product_args:
                del product_kwargs[key]

        # Produce instance of the produced Component
        instance = constructor(config, tool, **product_kwargs)
        return instance
