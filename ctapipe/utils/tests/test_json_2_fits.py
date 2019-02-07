"""A simple example of how to use traitlets.config.application.Application.

This should serve as a simple example that shows how the traitlets config
system works. The main classes are:

* traitlets.config.Configurable
* traitlets.config.SingletonConfigurable
* traitlets.config.Config
* traitlets.config.Application

To see the command line option help, run this program from the command line::

    $ python myapp.py -h

To make one of your classes configurable (from the command line and config
files) inherit from Configurable and declare class attributes as traits (see
classes Foo and Bar below). To make the traits configurable, you will need
to set the following options:

* ``config``: set to ``True`` to make the attribute configurable.
* ``shortname``: by default, configurable attributes are set using the syntax
  'Classname.attributename'. At the command line, this is a bit verbose, so
  we allow 'shortnames' to be declared. Setting a shortname is optional, but
  when you do this, you can set the option at the command line using the
  syntax: 'shortname=value'.
* ``help``: set the help string to display a help message when the ``-h``
  option is given at the command line. The help string should be valid ReST.

When the config attribute of an Application is updated, it will fire all of
the trait's events for all of the config=True attributes.
"""

from traitlets.config.configurable import Configurable
from traitlets.config.application import Application
from traitlets import (
    Bool, Unicode, Int, List, Dict
)

from ctapipe.utils.json2fits import traitlets_config_to_fits, json_to_fits
from ctapipe.utils import get_dataset_path
import tempfile

import sys
import os
import json


class Foo(Configurable):
    """A class that has configurable, typed attributes.

    """

    i = Int(0, help='The integer i.').tag(config=True)
    j = Int(1, help='The integer j.').tag(config=True)
    name = Unicode('Brian', help='First name.').tag(config=True)


class Bar(Configurable):

    enabled = Bool(True, help='Enable bar.').tag(config=True)


class MyApp(Application):

    name = Unicode('myapp')
    running = Bool(False,
                   help='Is the app running?').tag(config=True)
    classes = List([Bar, Foo])

    aliases = Dict(dict(
        i='Foo.i', j='Foo.j', name='Foo.name', running='MyApp.running',
        enabled='Bar.enabled', log_level='MyApp.log_lev',
        config_file='MyApp.config_file',
    ))

    flags = Dict(dict(
        enable=({'Bar': {'enabled': True}}, 'Enable Bar'),
        disable=({'Bar': {'enabled': False}}, 'Disable Bar'),
        debug=({'MyApp': {'log_level': 10}}, 'Set loglevel to DEBUG'),

    ))
    key_much_too_long_for_fits = Unicode('value short enough')
    key_and_value_too_long = Unicode(5 * 'value to long as wellenough')
    short = Unicode(20 * 'value to long')
    config_file = Unicode('', help='Load this config file').tag(config=True)

    def init_foo(self):
        # Pass config to other classes for them to inherit the config.
        self.foo = Foo(config=self.config)

    def init_bar(self):
        # Pass config to other classes for them to inherit the config.
        self.bar = Bar(config=self.config)

    def initialize(self, argv=None):
        self.parse_command_line(argv)
        self.full_path_configfile = self.config_file
        if self.config_file:
            self.load_config_file(self.config_file)
        self.init_foo()
        self.init_bar()

    def start(self):
        pass

    def stage_default_config_file(self):
        """auto generate default config file, and stage it into the profile."""
        # s = self.generate_config_file()
        fname = os.path.join('.', 'foo.json')  # self.config_file)
        if not os.path.exists(fname):
            self.log.warn(f'Generating default config file: {fname}')
            with open(fname, 'w') as f:
                f.write(str(json.dumps(self.config)))

    def traitlets_config_to_fits(self, outputfile):
        traitlets_config_to_fits(self.config, outputfile, overwrite=True)

    def json_to_fits(self, outputfile):
        json_to_fits(self.full_path_configfile, outputfile, overwrite=True)


def test_traitlets_config_to_fits():
    backup = sys.argv
    full_config_name = get_dataset_path('config.json')
    sys.argv = ['test_json_2_fits.py', '--config_f=' + full_config_name]
    app = MyApp()
    app.initialize()
    app.start()

    tmp = tempfile.NamedTemporaryFile()
    app.traitlets_config_to_fits(str(tmp.name))
    sys.argv = backup


def test_jsonToFits():
    backup = sys.argv
    full_config_name = get_dataset_path('config.json')
    sys.argv = ['test_json_2_fits.py', '--config_f=' + full_config_name]
    app = MyApp()
    app.initialize()
    app.start()
    tmp = tempfile.NamedTemporaryFile()
    app.json_to_fits(str(tmp.name))
    sys.argv = backup


def main():
    test_traitlets_config_to_fits()
    test_jsonToFits()


if __name__ == '__main__':
    main()
