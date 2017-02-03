from traitlets import Unicode
from traitlets.config import Application
from abc import abstractmethod
import logging

from ctapipe import __version__ as version


class ColoredFormatter(logging.Formatter):
    """
    Custom logging.Formatter that adds colors in addition to the original
    Application logger functionality from LevelFormatter (in application.py)
    """
    highlevel_limit = logging.WARN
    highlevel_format = " %(levelname)s |"

    def format(self, record):
        black, red, green, yellow, blue, magenta, cyan, white = range(8)
        reset_seq = "\033[0m"
        color_seq = "\033[1;%dm"
        colors = {
            'WARNING': yellow,
            'INFO': green,
            'DEBUG': blue,
            'CRITICAL': yellow,
            'ERROR': red
        }

        levelname = record.levelname
        if levelname in colors:
            levelname_color = color_seq % (30 + colors[levelname]) \
                              + levelname + reset_seq
            record.levelname = levelname_color

        if record.levelno >= self.highlevel_limit:
            record.highlevel = self.highlevel_format % record.__dict__
        else:
            record.highlevel = ""

        return super(ColoredFormatter, self).format(record)


class Tool(Application):
    """A base class for all executable tools (applications) that handles
    configuration loading/saving, logging, command-line processing,
    and provenance meta-data handling. It is based on
    `traitlets.config.Application`. Tools may contain configurable
    `ctapipe.core.Component` classes that do work, and their
    configuration parameters will propegate automatically to the
    `Tool`.

    Tool developers should create sub-classes, and a name,
    description, usage examples should be added by defining the
    `name`, `description` and `examples` class attributes as
    strings. The `aliases` attribute can be set to cause a lower-level
    `Component` parameter to become a high-plevel command-line
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

    config_file = Unicode(u'', help=("name of a configuration file with "
                                "parameters to load in addition to "
                                "command-line parameters")).tag(config=True)

    _log_formatter_cls = ColoredFormatter

    def __init__(self, **kwargs):
        # make sure there are some default aliases in all Tools:
        if self.aliases:
            self.aliases['log-level'] = 'Application.log_level'
            self.aliases['config'] = 'Tool.config_file'

        super().__init__(**kwargs)
        self.log_format = '%(levelname)8s [%(name)s]: %(message)s'
        self.log_level = 20  # default to INFO and above
        self.is_setup = False


    def initialize(self, argv=None):
        """ handle config and any other low-level setup """
        self.parse_command_line(argv)
        if self.config_file != '':
            self.log.debug("Loading config from '{}'".format(self.config_file))
            self.load_config_file(self.config_file)
        self.log.info("ctapipe version {}".format(self.version_string))
        self.setup()
        self.is_setup = True

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
        """
        try:
            self.initialize(argv)
            self.log.info("Starting: {}".format(self.name))
            self.log.debug("CONFIG: {}".format(self.config))
            self.start()
            self.finish()
        except ValueError as err:
            self.log.error('{}'.format(err))
        except RuntimeError as err:
            self.log.error('Caught unexpected exception: {}'.format(err))

    @property
    def version_string(self):
        """ a formatted version string with version, release, and git hash"""
        return "{}".format(version)
