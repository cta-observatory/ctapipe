""" Class to handle configuration for algorithms """
from abc import ABCMeta
from inspect import isabstract
from logging import getLogger

from traitlets import TraitError
from traitlets.config import Configurable


__all__ = ["non_abstract_children", "Component", "TelescopeComponent"]


def find_config_in_hierarchy(parent, class_name, trait_name):
    """
    Find the value of a config item in the hierarchy by going up the hierarchy
    from the parent and then down again to the child.
    This is needed as parent.config is the full config and not the
    config starting at the level of the parent.
    """

    config = parent.config

    # find the path from the config root to the desired object
    hierarchy = [class_name]
    while parent is not None:
        hierarchy.append(parent.__class__.__name__)
        parent = parent.parent

    hierarchy = list(reversed(hierarchy))

    # go down to the config value searched

    # root key is optional
    root = hierarchy.pop(0)
    if root in config:
        subconfig = config[root]
    else:
        subconfig = config

    for name in hierarchy:
        subconfig = subconfig[name]

    return subconfig[trait_name]


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
        g for s in base.__subclasses__() for g in non_abstract_children(s)
    ]
    non_abstract = [g for g in subclasses if not isabstract(g)]

    return non_abstract


class AbstractConfigurableMeta(type(Configurable), ABCMeta):
    """
    Metaclass to be able to make Component abstract
    see: http://stackoverflow.com/a/7314847/3838691
    """

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
    `self.log.info()`, `self.log.warning()`, `self.log.debug()`, etc).

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
                "Only one of `config` or `parent` allowed"
                " If you create a Component as part of another, give `parent=self`"
                " and not `config`"
            )
        super().__init__(parent=parent, config=config, **kwargs)

        for key, value in kwargs.items():
            if not self.has_trait(key):
                raise TraitError(f"Traitlet does not exist: {key}")

        # set up logging (for some reason the logger registered by LoggingConfig
        # doesn't use a child logger of the parent by default)
        if self.parent:
            self.log = self.parent.log.getChild(self.__class__.__name__)
        else:
            self.log = getLogger(
                self.__class__.__module__ + "." + self.__class__.__name__
            )

    @classmethod
    def from_name(cls, name, config=None, parent=None, **kwargs):
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
        parent : ctapipe.core.Tool
            Tool executable that is calling this component.
            Passes the correct logger and configuration to the component.
            This argument is typically only specified when using this method
            from within a Tool (config need not be passed if parent is used).
        kwargs

        Returns
        -------
        instace
            Instance of subclass to this class
        """
        requested_subclass = cls.non_abstract_subclasses()[name]
        return requested_subclass(config=config, parent=parent, **kwargs)

    @classmethod
    def non_abstract_subclasses(cls):
        """
        get dict{name: cls} of non abstract subclasses,
        subclasses can possibly be definded in plugins
        """
        subclasses = {base.__name__: base for base in non_abstract_children(cls)}
        return subclasses

    def get_current_config(self):
        """ return the current configuration as a dict (e.g. the values
        of all traits, even if they were not set during configuration)
        """
        name = self.__class__.__name__
        config = {name: {k: v.get(self) for k, v in self.traits(config=True).items()}}

        for val in self.__dict__.values():
            if isinstance(val, Component):
                config[name].update(val.get_current_config())

        return config

    def _repr_html_(self):
        """ nice HTML rep, with blue for non-default values"""
        traits = self.traits()
        name = self.__class__.__name__
        lines = [
            f"<b>{name}</b>",
            f"<p> {self.__class__.__doc__ or 'Undocumented!'} </p>",
            "<table>",
        ]
        for key, val in self.get_current_config()[name].items():
            # traits of the current component
            if key in traits:
                thehelp = f"{traits[key].help} (default: {traits[key].default_value})"
                lines.append(f"<tr><th>{key}</th>")
                if val != traits[key].default_value:
                    lines.append(f"<td><span style='color:blue'>{val}</span></td>")
                else:
                    lines.append(f"<td>{val}</td>")
                lines.append(f'<td style="text-align:left"><i>{thehelp}</i></td></tr>')
        lines.append("</table>")
        return "\n".join(lines)


class TelescopeComponent(Component):
    """
    A component that needs a SubarrayDescription to be constructed, and which
    contains configurable `TelescopeParameters` that must be configured on construction.
    """

    def __init__(self, subarray, config=None, parent=None, **kwargs):
        super().__init__(config, parent, **kwargs)

        self.subarray = subarray
        # configure all of the TelescopeParameters
        for trait in list(self.class_traits()):
            try:
                getattr(self, trait).attach_subarray(subarray)
            except (AttributeError, TypeError):
                pass

    @classmethod
    def from_name(cls, name, subarray, config=None, parent=None, **kwargs):
        """
        Obtain an instance of a subclass via its name

        Parameters
        ----------
        name : str
            Name of the subclass to obtain
        subarray: ctapipe.instrument.SubarrayDescription
            The current subarray for this TelescopeComponent.
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            This argument is typically only specified when using this method
            from within a Tool.
        parent : ctapipe.core.Tool
            Tool executable that is calling this component.
            Passes the correct logger and configuration to the component.
            This argument is typically only specified when using this method
            from within a Tool (config need not be passed if parent is used).
        kwargs

        Returns
        -------
        instace
            Instance of subclass to this class
        """
        requested_subclass = cls.non_abstract_subclasses()[name]
        return requested_subclass(
            subarray=subarray, config=config, parent=parent, **kwargs
        )
