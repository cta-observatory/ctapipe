""" Class to handle configuration for algorithms """
from abc import ABCMeta
from logging import getLogger
from inspect import isabstract
from traitlets.config import Configurable
from traitlets import TraitError
from ctapipe.core.plugins import detect_and_import_io_plugins


def non_abstract_children(base):
    """
    Return all non-abstract subclasses of a base class recursively.

    Parameters
    ----------
    base : class
        High level class object that is inherited by the
        desired subclasses
    Returns
    -------
    non_abstract : dict
        dict of all non-abstract subclasses
     """
    subclasses = base.__subclasses__() + [
        g for s in base.__subclasses__()
        for g in non_abstract_children(s)
    ]
    non_abstract = [g for g in subclasses if not isabstract(g)]

    return non_abstract


class AbstractConfigurableMeta(type(Configurable), ABCMeta):
    '''
    Metaclass to be able to make Component abstract
    see: http://stackoverflow.com/a/7314847/3838691
    '''
    pass


class Component(Configurable, metaclass=AbstractConfigurableMeta):
    """Base class of all Components.

    Components are classes that are configurable via traitlets
    and setup a logger in the ctapipe logging hierarchy.

    `traitlets` can validate values and provide defaults and
    descriptions. These will be automatically translated into
    configuration parameters (command-line, config file, etc). Note
    that any parameter that should be externally configurable must
    have its `config` attribute set to `True`, e.g. defined like
    `myparam = Integer(0, help='the parameter').tag(config=True)`.

    All components also contain a `logger` instance in their `log`
    attribute, that you must use to output info, debugging data,
    warnings, etc (do not use `print()` statements, instead use
    `self.log.info()`, `self.log.warn()`, `self.log.debug()`, etc).

    Components are generally used within `ctapipe.core.Tool`
    subclasses, which provide configuration handling and command-line
    tool generation.

    For example:

    .. code:: python

        from ctapipe.core import Component
        from traitlets import (Integer, Float)

        class MyComponent(Component):
            \"\"\" Does something \"\"\"
            some_option = Integer(default_value=6,
                                  help='a value to set').tag(config=True)


        comp = MyComponent()
        comp.some_option = 6      # ok
        comp.some_option = 'test' # will fail validation
    """

    def __init__(self, config=None, parent=None, **kwargs):
        """
        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
        parent: Tool or Component
            If a Component is created by another Component or Tool,
            you need to pass the creating Component as parent, e.g.
            `parent=self`. This makes sure the config is correctly
            handed down to the child components.
            Do not pass config in this case.
        kwargs
            Traitlets to be overridden.
            TraitError is raised if kwargs contains a key that does not
            correspond to a traitlet.
        """
        if parent is not None and config is not None:
            raise ValueError(
                'Only one of `config` or `parent` allowed'
                ' If you create a Component as part of another, give `parent=self`'
                ' and not `config`'
            )
        super().__init__(parent=parent, config=config, **kwargs)

        for key, value in kwargs.items():
            if not self.has_trait(key):
                raise TraitError(f"Traitlet does not exist: {key}")

        # set up logging
        if self.parent:
            self.log = self.parent.log.getChild(self.__class__.__name__)
        else:
            self.log = getLogger(
                self.__class__.__module__ + '.' + self.__class__.__name__
            )

    @classmethod
    def from_name(cls, name, config=None, parent=None):
        """
        Obtain an instance of a subclass via its name

        Parameters
        ----------
        name : str
            Name of the subclass to obtain
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            This argument is typically only specified when using this method
            from within a Tool.
        tool : ctapipe.core.Tool
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            This argument is typically only specified when using this method
            from within a Tool.

        Returns
        -------
        instace
            Instance of subclass to this class
        """
        detect_and_import_io_plugins()
        subclasses = {
            base.__name__: base
            for base in non_abstract_children(cls)
        }
        requested_subclass = subclasses[name]

        return requested_subclass(config=config, parent=parent)
