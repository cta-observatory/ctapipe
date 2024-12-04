"""Classes to handle configurable command-line user interfaces."""

import html
import logging
import logging.config
import os
import pathlib
import re
import textwrap
from abc import abstractmethod
from contextlib import ExitStack
from inspect import cleandoc
from subprocess import CalledProcessError
from tempfile import mkdtemp

import yaml
from docutils.core import publish_parts
from traitlets import TraitError

try:
    import tomli as toml

    HAS_TOML = True
except ImportError:
    HAS_TOML = False

from traitlets import List, default
from traitlets.config import Application, Config, Configurable

from .. import __version__ as version
from . import Provenance
from .component import Component
from .logging import ColoredFormatter, create_logging_config
from .traits import Bool, Dict, Enum, Path

__all__ = ["Tool", "ToolConfigurationError"]


class CollectTraitWarningsHandler(logging.NullHandler):
    regex = re.compile(".*Config option.*not recognized")

    def __init__(self):
        super().__init__()
        self.errors = []

    def handle(self, record):
        if self.regex.match(record.msg) and record.levelno == logging.WARNING:
            self.errors.append(record.msg)


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
    ``name``, ``description`` and ``examples`` class attributes as
    strings. The ``aliases`` attribute can be set to cause a lower-level
    `~ctapipe.core.Component` parameter to become a high-level command-line
    parameter (See example below). The `setup`, `start`, and
    `finish` methods should be defined in the sub-class.

    Additionally, any `ctapipe.core.Component` used within the `Tool`
    should have their class in a list in the ``classes`` attribute,
    which will automatically add their configuration parameters to the
    tool.

    Once a tool is constructed and the abstract methods are implemented,
    the user can call the `run` method to setup and start it.

    Tools have an `~contextlib.ExitStack` to guarantee cleanup tasks are
    run when the tool terminates, also in case of errors. If a task needs
    a cleanup, it must be a context manager and ``Tool.enter_context``
    must be called on the object. This will guarantee that the ``__exit__``
    method of the context manager is called after the tool has finished
    executing. This happens after the ``finish`` method has run or
    in case of exceptions.

    User-defined code can raise custom exceptions both in the components
    or in the tool methods. If these custom exceptions have an ``exit_code`` attribute,
    it will be propagated to the final exit code of the tool.

    .. code:: python

        from ctapipe.core import Tool
        from traitlets import (Integer, Float, Dict, Unicode)

        class MyTool(Tool):
            name = "mytool"
            description = "do some things and stuff"
            aliases = Dict({'infile': 'AdvancedComponent.infile',
                            'iterations': 'MyTool.iterations'})

            # Which classes are registered for configuration
            classes = [MyComponent, AdvancedComponent, SecondaryMyComponent]

            # local configuration parameters
            iterations = Integer(5,help="Number of times to run",
                                 allow_none=False).tag(config=True)

            def setup(self):
                self.comp = MyComponent(self, parent=self)
                self.comp2 = SecondaryMyComponent(self, parent=self)

                # correct use of component that is a context manager
                # using it like this makes sure __exit__ will be called
                # at the end of Tool.run, even in case of exceptions
                self.comp3 = self.enter_context(MyComponent(parent=self))

                self.advanced = AdvancedComponent(parent=self)

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


    If this ``main()`` function is registered in ``setup.py`` under
    *entry_points*, it will become a command-line tool (see examples
    in the ``ctapipe/tools`` subdirectory).

    """

    config_files = List(
        trait=Path(
            exists=True,
            directory_ok=False,
        ),
        help=(
            "List of configuration files with parameters to load "
            "in addition to command-line parameters. "
            "The order listed is the order of precedence (later config parameters "
            "overwrite earlier ones), however parameters specified on the "
            "command line always have the highest precedence. "
            "Config files may be in JSON, YAML, TOML, or Python format"
        ),
    ).tag(config=True)

    log_config = Dict(default_value={}).tag(config=True)

    log_file = Path(
        default_value=None,
        exists=None,
        directory_ok=False,
        help="Filename for the log",
        allow_none=True,
    ).tag(config=True)

    log_file_level = Enum(
        values=Application.log_level.values,
        default_value="INFO",
        help="Logging Level for File Logging",
    ).tag(config=True)

    quiet = Bool(default_value=False).tag(config=True)
    overwrite = Bool(default_value=False).tag(config=True)

    _log_formatter_cls = ColoredFormatter

    provenance_log = Path(directory_ok=False).tag(config=True)

    @default("provenance_log")
    def _default_provenance_log(self):
        return self.name + ".provenance.log"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # make sure there are some default aliases in all Tools:
        aliases = {
            ("c", "config"): "Tool.config_files",
            "log-level": "Tool.log_level",
            ("l", "log-file"): "Tool.log_file",
            "log-file-level": "Tool.log_file_level",
            "provenance-log": "Tool.provenance_log",
        }
        # makes sure user defined aliases override default aliases
        self.aliases = {**aliases, **self.aliases}

        flags = {
            ("q", "quiet"): ({"Tool": {"quiet": True}}, "Disable console logging."),
            ("v", "verbose"): (
                {"Tool": {"log_level": "DEBUG"}},
                "Set log level to DEBUG",
            ),
            "overwrite": (
                {"Tool": {"overwrite": True}},
                "Overwrite existing output files without asking",
            ),
        }
        self.flags = {**flags, **self.flags}

        self.is_setup = False
        self.version = version
        self.raise_config_file_errors = True  # override traitlets.Application default

        # tools defined in other modules should have those modules as base
        # logging name
        self.module_name = self.__class__.__module__.split(".")[0]
        self.log = logging.getLogger(f"{self.module_name}.{self.name}")
        self.trait_warning_handler = CollectTraitWarningsHandler()
        self.update_logging_config()
        self._exit_stack = ExitStack()

    def enter_context(self, context_manager):
        """
        Add a new context manager to the `~contextlib.ExitStack` of this Tool

        This method should be used with things that need a cleanup step,
        also in case of exception. ``enter_context`` will call
        ``context_manager.__enter__`` and return its result.

        This will guarantee that ``context_manager.__exit__`` is called when
        `~ctapipe.core.Tool.run` finishes, even when an error occurs.
        """
        return self._exit_stack.enter_context(context_manager)

    def initialize(self, argv=None):
        """handle config and any other low-level setup"""
        self.parse_command_line(argv)
        self.update_logging_config()

        if self.config_files is not None:
            self.log.info("Loading config from '%s'", self.config_files)
            try:
                for config_file in self.config_files:
                    self.load_config_file(config_file)
            except Exception as err:
                raise ToolConfigurationError(
                    f"Couldn't read config file: {err} ({type(err)})"
                ) from err

        # ensure command-line takes precedence over config file options:
        self.update_config(self.cli_config)
        self.update_logging_config()

        self.log.info("ctapipe version %s", self.version_string)

    def load_config_file(self, path: str | pathlib.Path) -> None:
        """
        Load a configuration file in one of the supported formats, and merge it with
        the current config if it exists.

        Parameters
        ----------
        path: Union[str, pathlib.Path]
            config file to load. [yaml, toml, json, py] formats are supported
        """
        path = pathlib.Path(path)

        if path.suffix in [".yaml", ".yml"]:
            # do our own YAML loading
            with open(path) as infile:
                config = Config(yaml.safe_load(infile))
            self.update_config(config)
        elif path.suffix == ".toml" and HAS_TOML:
            with open(path, "rb") as infile:
                config = Config(toml.load(infile))
            self.update_config(config)
        elif path.suffix in [".json", ".py"]:
            # fall back to traitlets.config.Application's implementation. Note
            # that if we don't specify the file suffixes here, traitlets seems
            # to silently ignore unknown ones.
            super().load_config_file(str(path))
        else:
            raise ToolConfigurationError(
                f"The config file '{path}' is not in a known format. "
                "The file should end in one: yml, yaml, toml, json, py"
            )

        Provenance().add_input_file(path, role="Tool Configuration", add_meta=False)

    def update_logging_config(self):
        """Update the configuration of loggers."""
        cfg = create_logging_config(
            log_level=self.log_level,
            log_file=self.log_file,
            log_file_level=self.log_file_level,
            log_config=self.log_config,
            quiet=self.quiet,
            module=self.module_name,
        )

        logging.config.dictConfig(cfg)

        # re-add our custom handler every time the config is updated.
        self.log.addHandler(self.trait_warning_handler)

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

    def check_output(self, *output_paths):
        """
        Test if output files exist and if they do, throw an error
        unless ``self.overwrite`` is set to True.
        This should be checked during tool setup to avoid having a tool only
        realize the output can not be written after some long-running calculations
        (e.g. training of ML-models).
        Because we currently do not collect all created output files in the tool
        (they can be attached to some component), the output files need
        to be given and can not easily be derived from ``self``.

        Parameters
        ----------
        output_paths: Path
            One or more output path to check.

        """
        for output in output_paths:
            if output is not None and output.exists():
                if self.overwrite:
                    self.log.warning("Overwriting %s", output)
                else:
                    raise ToolConfigurationError(
                        f"Output path {output} exists, but overwrite=False"
                    )

    @abstractmethod
    def setup(self):
        """Set up the tool.

        This method runs after the configuration and command line options
        have been parsed.

        Here the tool should construct all ``Components``, open files, etc.
        """

    @abstractmethod
    def start(self):
        """
        Main function of the tool.

        This is automatically called after `Tool.initialize` when `Tool.run` is called.
        """

    @abstractmethod
    def finish(self):
        """
        Finish up.

        This is called automatically after `Tool.start` when `Tool.run` is called.
        """
        self.log.info("Goodbye")

    def run(self, argv=None, raises=False):
        """Run the tool.

        This automatically calls `Tool.initialize`, `Tool.start` and `Tool.finish`

        Parameters
        ----------

        argv: list(str)
            command-line arguments, or None to get them
            from sys.argv automatically

        raises : bool
            Whether to raise Exceptions (to test them) or not.
        """

        # return codes are taken from:
        #  https://tldp.org/LDP/abs/html/exitcodes.html

        status = "completed"
        exit_status = 0
        current_exception = None

        with self._exit_stack:
            try:
                self.log.info("Starting: %s", self.name)
                Provenance().start_activity(self.name)

                self.initialize(argv)

                self.setup()
                self.is_setup = True

                self.log.debug("CONFIG: %s", self.get_current_config())
                Provenance().add_config(self.get_current_config())

                # check for any traitlets warnings using our custom handler
                if len(self.trait_warning_handler.errors) > 0:
                    raise ToolConfigurationError("Found config errors")

                # remove handler to not impact performance with regex matching
                self.log.removeHandler(self.trait_warning_handler)

                self.start()
                self.finish()
            except (ToolConfigurationError, TraitError) as err:
                current_exception = err
                self.log.error("%s", err)
                self.log.error("Use --help for more info")
                exit_status = 2  # wrong cmd line parameter
                status = "error"
            except KeyboardInterrupt:
                self.log.warning("WAS INTERRUPTED BY CTRL-C")
                exit_status = 130  # Script terminated by Control-C
                status = "interrupted"
            except Exception as err:
                current_exception = err
                exit_status = getattr(err, "exit_code", 1)
                status = "error"
                self.log.exception("Caught unexpected exception: %s", err)
            except SystemExit as err:
                exit_status = err.code
                if exit_status != 0:
                    status = "error"
                    current_exception = err
                    self.log.critical(
                        "Caught SystemExit with exit code %s", exit_status
                    )
            finally:
                if raises and current_exception:
                    self.write_provenance()
                    raise current_exception

        Provenance().finish_activity(
            activity_name=self.name, status=status, exit_code=exit_status
        )

        if not {"-h", "--help", "--help-all"}.intersection(self.argv):
            self.write_provenance()

        self.log.info("Finished %s", self.name)
        self.exit(exit_status)

    def write_provenance(self):
        for activity in Provenance().finished_activities:
            output_str = " ".join([x["url"] for x in activity.output])
            self.log.info("Output: %s", output_str)

        self.log.debug(
            "PROVENANCE: 'Details about provenance is found in %s'", self.provenance_log
        )
        self.provenance_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.provenance_log, mode="a+") as provlog:
            provlog.write(Provenance().as_json(indent=3))

    @property
    def version_string(self):
        """a formatted version string with version, release, and git hash"""
        return f"{version}"

    def get_current_config(self):
        """return the current configuration as a dict (e.g. the values
        of all traits, even if they were not set during configuration)
        """
        conf = {
            self.__class__.__name__: {
                k: v.get(self) for k, v in self.traits(config=True).items()
            }
        }

        for val in self.__dict__.values():
            if isinstance(val, Component):
                conf[self.__class__.__name__].update(val.get_current_config())

        return conf

    def _repr_html_(self):
        """nice HTML rep, with blue for non-default values"""
        traits = self.traits()
        name = self.__class__.__name__
        docstring = (
            publish_parts(
                cleandoc(self.__class__.__doc__ or self.description), writer_name="html"
            )["html_body"]
            or "Undocumented"
        )
        lines = [
            '<div style="border:1px solid black; max-width: 700px; padding:2em; word-wrap:break-word;">',
            f"<b>{name}</b>",
            docstring,
            "<table>",
            "    <colgroup>",
            "        <col span='1' style=' '>",
            "        <col span='1' style='width: 20em;'>",
            "        <col span='1' >",
            "    </colgroup>",
            "    <tbody>",
        ]
        for key, val in self.get_current_config()[name].items():
            htmlval = (
                html.escape(str(val)).replace("/", "/<wbr>").replace("_", "_<wbr>")
            )  # allow breaking at boundary

            # traits of the current component
            if key in traits:
                thehelp = html.escape(
                    f"{traits[key].help} (default: {traits[key].default_value})"
                )
                lines.append(f"<tr><th>{key}</th>")
                if val != traits[key].default_value:
                    lines.append(
                        f"<td style='text-align: left;'><span style='color:blue; max-width:30em;'>{htmlval}</span></td>"
                    )
                else:
                    lines.append(f"<td style='text-align: left;'>{htmlval}</td>")
                lines.append(
                    f"<td style='text-align: left;'><i>{thehelp}</i></td></tr>"
                )
        lines.append("    </tbody>")
        lines.append("</table>")
        lines.append("<p><b>Components:</b>")
        lines.append(", ".join([x.__name__ for x in self.classes]))
        lines.append("</p>")
        lines.append("</div>")

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
    breaker = "#" + "-" * 78
    parent_classes = ", ".join(
        p.__name__ for p in tool.__bases__ if issubclass(p, Configurable)
    )

    section_header = f"# {tool.__name__}({parent_classes}) configuration"

    lines = [breaker, section_header]
    # get the description trait
    desc = tool.class_traits().get("description")
    if desc:
        desc = desc.default_value
    if not desc:
        # no description from trait, use __doc__
        desc = getattr(tool, "__doc__", "")
    if desc:
        lines.append(commented(desc, indent_level=0))
    lines.append(breaker)
    lines.append(f"{tool.__name__}:")

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
            if "Enum" in type(trait).__name__:
                # include Enum choices
                lines.append(commented(f"Choices: {trait.info()}"))
            lines.append(commented(f"Default: {default_repr}"))
        else:
            # Trait appears multiple times and isn't defined here.
            # Truncate help to first line + "See also Original.trait"
            if trait.help:
                lines.append(commented(trait.help.split("\n", 1)[0]))
            lines.append(f"    # See also: {defining_class.__name__}.{name}")
        lines.append(f"    {name}: {current_repr}")
        lines.append("")
    return "\n".join(lines)


def run_tool(tool: Tool, argv=None, cwd=None, raises=True):
    """
    Utility run a certain tool in a python session without exiting.

    Parameters
    ----------
    argv : List[str]
        List of command line arguments for the tool.
    cwd : str or pathlib.Path
        Path to a temporary working directory. If none, a new (random)
        temporary directeory gets created.
    raises : bool
        If true, raises Exceptions from running tools, to test them.
        If false, tools can return a non-zero exit code.

    Returns
    -------
    exit_code: int
        The return code of the tool, 0 indicates success, everything else an error
    """
    current_cwd = pathlib.Path().absolute()
    cwd = pathlib.Path(cwd) if cwd is not None else mkdtemp()
    argv = argv or []
    try:
        # switch to cwd for running and back after
        os.chdir(cwd)
        tool.run(argv, raises=raises)
    except SystemExit as e:
        if raises and e.code != 0:
            raise CalledProcessError(e.code, [tool.name] + argv)
        return e.code
    finally:
        os.chdir(current_cwd)
