import logging
from abc import abstractmethod

from traitlets import Unicode
from traitlets.config import Application

from ctapipe import __version__ as version
from .logging import ColoredFormatter
from . import Provenance

logging.basicConfig(level=logging.WARNING)


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

    _log_formatter_cls = ColoredFormatter

    def __init__(self, **kwargs):
        # make sure there are some default aliases in all Tools:
        if self.aliases:
            self.aliases['log-level'] = 'Application.log_level'
            self.aliases['config'] = 'Tool.config_file'

        super().__init__(**kwargs)
        self.log_format = ('%(levelname)8s [%(name)s] '
                           '(%(module)s/%(funcName)s): %(message)s')
        self.log_level = logging.INFO
        self.is_setup = False

    def initialize(self, argv=None):
        """ handle config and any other low-level setup """
        self.parse_command_line(argv)
        if self.config_file != '':
            self.log.debug(f"Loading config from '{self.config_file}'")
            self.load_config_file(self.config_file)
        self.log.info(f"ctapipe version {self.version_string}")

    @abstractmethod
    def setup(self):
        """set up the tool (override in subclass). Here the user should
        construct all `Components` and open files, etc."""
        pass

    @abstractmethod
    def start(self):
        """main body of tool (override in subclass). This is automatially
        called after `initialize()` when the `run()` is called.
        """
        pass

    @abstractmethod
    def finish(self):
        """finish up (override in subclass). This is called automatially
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
            self.log.debug(f"CONFIG: {self.config}")
            Provenance().start_activity(self.name)
            Provenance().add_config(self.config)
            self.setup()
            self.is_setup = True
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
