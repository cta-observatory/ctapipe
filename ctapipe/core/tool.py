""" Classes to handle configurable command-line user interfaces """
import logging
import textwrap
from abc import abstractmethod

from traitlets import Unicode
from traitlets.config import Application, Configurable

from ctapipe import __version__ as version
from . import Provenance
from .logging import ColoredFormatter


class ToolConfigurationError(Exception):

    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        self.message = message


class Tool(Application):
    """A base class for all executable tools (applications) that handles
    configuration loading/saving, logging, command-line processing,
    and provenance meta-data handling. It is based on
    `traitlets.config.Application`. Tools may contain configurable
    `ctapipe.core.Component` classes that do work, and their
    configuration parameters will propagate automatically to the
    `Tool`.

    Tool developers should create sub-classes, and a name,
    description, usage examples should be added by defining the
    `name`, `description` and `examples` class attributes as
    strings. The `aliases` attribute can be set to cause a lower-level
    `Component` parameter to become a high-level command-line
    parameter (See example below). The `setup()`, `start()`, and
    `finish()` methods should be defined in the sub-class.

    Additionally, any `ctapipe.core.Component` used within the `Tool`
    should have their class in a list in the `classes` attribute,
    which will automatically add their configuration parameters to the
    tool.

    Once a tool is constructed and the virtual methods defined, the
    user can call the `run()` method to setup and start it.


    .. code:: python

        from ctapipe.core import Tool
        from traitlets import (Integer, Float, List, Dict, Unicode)

        class MyTool(Tool):
            name = "mytool"
            description = "do some things and stuff"
            aliases = Dict({'infile': 'AdvancedComponent.infile',
                            'iterations': 'MyTool.iterations'})

            # Which classes are registered for configuration
            classes = List([MyComponent, AdvancedComponent,
                            SecondaryMyComponent])

            # local configuration parameters
            iterations = Integer(5,help="Number of times to run",
                                 allow_none=False).tag(config=True)

            def setup_comp(self):
                self.comp = MyComponent(self, config=self.config)
                self.comp2 = SecondaryMyComponent(self, config=self.config)

            def setup_advanced(self):
                self.advanced = AdvancedComponent(self, config=self.config)

            def setup(self):
                self.setup_comp()
                self.setup_advanced()

            def start(self):
                self.log.info("Performing {} iterations..."\
                              .format(self.iterations))
                for ii in range(self.iterations):
                    self.log.info("ITERATION {}".format(ii))
                    self.comp.do_thing()
                    self.comp2.do_thing()
                    sleep(0.5)

            def finish(self):
                self.log.warning("Shutting down.")

        def main():
            tool = MyTool()
            tool.run()

        if __name__ == "main":
           main()


    If this `main()` method is registered in `setup.py` under
    *entry_points*, it will become a command-line tool (see examples
    in the `ctapipe/tools` subdirectory).

    """

    config_file = Unicode('', help=("name of a configuration file with "
                                    "parameters to load in addition to "
                                    "command-line parameters")).tag(config=True)
    log_format = Unicode(
        '%(levelname)s [%(name)s] (%(module)s/%(funcName)s): %(message)s',
        help='The Logging format template'
    ).tag(config=True)

    _log_formatter_cls = ColoredFormatter

    def __init__(self, **kwargs):
        # make sure there are some default aliases in all Tools:
        if self.aliases:
            self.aliases['log-level'] = 'Application.log_level'
            self.aliases['config'] = 'Tool.config_file'

        super().__init__(**kwargs)
        self.log_level = logging.INFO
        self.is_setup = False
        self._registered_components = []

    def initialize(self, argv=None):
        """ handle config and any other low-level setup """
        self.parse_command_line(argv)
        if self.config_file != '':
            self.log.debug(f"Loading config from '{self.config_file}'")
            self.load_config_file(self.config_file)
        self.log.info(f"ctapipe version {self.version_string}")

    def add_component(self, component_instance):
        """
        constructs and adds a component to the list of registered components,
        so that later we can ask for the current configuration of all instances,
        e.g. in`get_full_config()`.  All sub-components of a tool should be
        constructed using this function, in order to ensure the configuration is
        properly traced.

        Parameters
        ----------
        component_instance: Component
            constructed instance of a component

        Returns
        -------
        Component:
            the same component instance that was passed in, so that the call
            can be chained.

        Examples
        --------
        .. code-block:: python3

            self.mycomp = self.add_component(MyComponent(parent=self))

        """
        self._registered_components.append(component_instance)
        return component_instance

    @abstractmethod
    def setup(self):
        """set up the tool (override in subclass). Here the user should
        construct all `Components` and open files, etc."""
        pass

    @abstractmethod
    def start(self):
        """main body of tool (override in subclass). This is  automatically
        called after `initialize()` when the `run()` is called.
        """
        pass

    @abstractmethod
    def finish(self):
        """finish up (override in subclass). This is called automatically
        after `start()` when `run()` is called."""
        self.log.info("Goodbye")

    def run(self, argv=None):
        """Run the tool. This automatically calls `initialize()`,
        `start()` and `finish()`

        Parameters
        ----------

        argv: list(str)
            command-line arguments, or None to get them
            from sys.argv automatically
        """
        try:
            self.initialize(argv)
            self.log.info(f"Starting: {self.name}")
            Provenance().start_activity(self.name)
            self.setup()
            self.is_setup = True
            self.log.info(f"CONFIG: {self.get_current_config()}")
            Provenance().add_config(self.get_current_config())
            self.start()
            self.finish()
            self.log.info(f"Finished: {self.name}")
            Provenance().finish_activity(activity_name=self.name)
        except ToolConfigurationError as err:
            self.log.error(f'{err}.  Use --help for more info')
        except RuntimeError as err:
            self.log.error(f'Caught unexpected exception: {err}')
            self.finish()
            Provenance().finish_activity(activity_name=self.name,
                                         status='error')
        except KeyboardInterrupt:
            self.log.warning("WAS INTERRUPTED BY CTRL-C")
            self.finish()
            Provenance().finish_activity(activity_name=self.name,
                                         status='interrupted')
        finally:
            for activity in Provenance().finished_activities:
                output_str = ' '.join([x['url'] for x in activity.output])
                self.log.info("Output: %s", output_str)

            self.log.debug("PROVENANCE: '%s'", Provenance().as_json(indent=3))

    @property
    def version_string(self):
        """ a formatted version string with version, release, and git hash"""
        return f"{version}"

    def get_current_config(self):
        """ return the current configuration as a dict (e.g. the values
        of all traits, even if they were not set during configuration)
        """
        conf = {
            self.__class__.__name__: {
                k: v.get(self) for k, v in self.traits(config=True).items()
            }
        }
        for component in self._registered_components:
            conf.update(component.get_current_config())

        return conf

    def _repr_html_(self):
        """ nice HTML rep, with blue for non-default values"""
        traits = self.traits()
        name = self.__class__.__name__
        lines = [
            f"<b>{name}</b>",
            f"<p> {self.__class__.__doc__ or self.description} </p>",
            "<table>",
        ]
        for key, val in self.get_current_config()[name].items():
            default = traits[key].default_value
            thehelp = f'{traits[key].help} (default: {default})'
            lines.append(f"<tr><th>{key}</th>")
            if val != default:
                lines.append(f"<td><span style='color:blue'>{val}</span></td>")
            else:
                lines.append(f"<td>{val}</td>")
            lines.append(f'<td style="text-align:left"><i>{thehelp}</i></td></tr>')
        lines.append("</table>")
        lines.append("<p><i>Components:</i>")
        lines.append(", ".join([x.__name__ for x in self.classes]))
        lines.append("</p>")

        return "\n".join(lines)


def export_tool_config_to_commented_yaml(tool_instance: Tool, classes=None):
    """
    Turn the config of a single Component into a commented YAML string.

    This is a hacked version of
    traitlets.config.Configurable._class_config_section() changed to
    output a  YAML file with defaults *and* current values filled in.

    Parameters
    ----------
    tool_instance: Tool
        a constructed Tool instance
    classes: list, optional
        The list of other classes in the config file.
        Used to reduce redundant information.
    """

    tool = tool_instance.__class__
    config = tool_instance.get_current_config()[tool_instance.__class__.__name__]

    def commented(text, indent_level=2, width=70):
        """return a commented, wrapped block."""
        return textwrap.fill(
            text,
            width=width,
            initial_indent="  " * indent_level + "# ",
            subsequent_indent="  " * indent_level + "# ",
        )

    # section header
    breaker = '#' + '-' * 78
    parent_classes = ', '.join(
        p.__name__ for p in tool.__bases__
        if issubclass(p, Configurable)
    )

    section_header = f"# {tool.__name__}({parent_classes}) configuration"

    lines = [breaker, section_header]
    # get the description trait
    desc = tool.class_traits().get('description')
    if desc:
        desc = desc.default_value
    if not desc:
        # no description from trait, use __doc__
        desc = getattr(tool, '__doc__', '')
    if desc:
        lines.append(commented(desc, indent_level=0))
    lines.append(breaker)
    lines.append(f'{tool.__name__}:')

    for name, trait in sorted(tool.class_traits(config=True).items()):
        default_repr = trait.default_value_repr()
        current_repr = config.get(name, "")
        if isinstance(current_repr, str):
            current_repr = f'"{current_repr}"'

        if classes:
            defining_class = tool._defining_class(trait, classes)
        else:
            defining_class = tool
        if defining_class is tool:
            # cls owns the trait, show full help
            if trait.help:
                lines.append(commented(trait.help))
            if 'Enum' in type(trait).__name__:
                # include Enum choices
                lines.append(commented(f'Choices: {trait.info()}'))
            lines.append(commented(f'Default: {default_repr}'))
        else:
            # Trait appears multiple times and isn't defined here.
            # Truncate help to first line + "See also Original.trait"
            if trait.help:
                lines.append(commented(trait.help.split('\n', 1)[0]))
            lines.append(
                f'    # See also: {defining_class.__name__}.{name}')
        lines.append(f'    {name}: {current_repr}')
        lines.append('')
    return '\n'.join(lines)
