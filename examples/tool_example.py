"""A simple example of how to use traitlets.config.application.Application.
This should serve as a simple example that shows how the traitlets config
system works. The main classes are:
* traitlets.config.Configurable
* traitlets.config.SingletonConfigurable
* traitlets.config.Config
* traitlets.config.Application
To see the command line option help, run this program from the command line::
    $ python test_tool.py --help
To make one of your classes configurable (from the command line and config
files) inherit from Configurable and declare class attributes as traits (see
classes Foo and Bar below). To make the traits configurable, you will need
to set the following options:
* ``config``: set to ``True`` to make the attribute configurable.
* ``shortname``: by default, configurable attributes are set using the syntax
  "Classname.attributename". At the command line, this is a bit verbose, so
  we allow "shortnames" to be declared. Setting a shortname is optional, but
  when you do this, you can set the option at the command line using the
  syntax: "shortname=value".
* ``help``: set the help string to display a help message when the ``-h``
  option is given at the command line. The help string should be valid ReST.
When the config attribute of an Application is updated, it will fire all of
the trait's events for all of the config=True attributes.
"""

from traitlets import Bool, Unicode, Int, List, Dict

from ctapipe.core import Component, Tool


class AComponent(Component):
    """
    A class that has configurable, typed attributes.
    """

    i = Int(0, help="The integer i.").tag(config=True)
    j = Int(1, help="The integer j.").tag(config=True)
    name = Unicode("Brian", help="First name.").tag(config=True)

    def __call__(self):
        self.log.info("CALLED FOO")


class BComponent(Component):
    """ Some Other Component """

    enabled = Bool(True, help="Enable bar.").tag(config=True)


class MyTool(Tool):
    """ My Tool """

    name = Unicode("myapp")
    running = Bool(False, help="Is the app running?").tag(config=True)
    classes = List([BComponent, AComponent])
    config_file = Unicode("", help="Load this config file").tag(config=True)

    aliases = Dict(
        dict(
            i="Foo.i",
            j="Foo.j",
            name="Foo.name",
            running="MyApp.running",
            enabled="Bar.enabled",
            log_level="MyApp.log_level",
        )
    )

    flags = Dict(
        dict(
            enable=({"Bar": {"enabled": True}}, "Enable Bar"),
            disable=({"Bar": {"enabled": False}}, "Disable Bar"),
            debug=({"MyApp": {"log_level": 10}}, "Set loglevel to DEBUG"),
        )
    )

    def init_a_component(self):
        """ setup the Foo component"""
        self.log.info("INIT FOO")
        self.a_component = self.add_component(AComponent(parent=self))

    def init_b_component(self):
        """ setup the Bar component"""
        self.log.info("INIT BAR")
        self.b_component = self.add_component(BComponent(parent=self))

    def setup(self):
        """ Setup all components and the tool"""
        self.init_a_component()
        self.init_b_component()

    def start(self):
        """ run the tool"""
        self.log.info("app.config:")
        self.log.info("THE CONFIGURATION: %s", self.get_current_config())
        self.a_component()


def main():
    """ run the app """
    tool = MyTool()
    tool.run()


if __name__ == "__main__":
    main()
