from ctapipe.core import Tool
from traitlets.config import KVArgParseConfigLoader
from traitlets import Dict
from copy import copy, deepcopy
import sys
from ipython_genutils import py3compat
import re
from traitlets.config.application import catch_config_error


class CustomLoader(KVArgParseConfigLoader):
    """
    Identical to KVArgParseConfigLoader, except self.extra_args are not parsed,
    and therefore no error is thrown by command line arguments that are only
    relevant after factories have been executed.
    """
    def _convert_to_config(self):
        """self.parsed_data->self.config, parse unrecognized extra args
        via KVLoader."""
        # remove subconfigs list from namespace before transforming the
        # Namespace
        if '_flags' in self.parsed_data:
            subcs = self.parsed_data._flags
            del self.parsed_data._flags
        else:
            subcs = []

        for k, v in vars(self.parsed_data).items():
            if v is None:
                # it was a flag that shares the name of an alias
                subcs.append(self.alias_flags[k])
            else:
                # eval the KV assignment
                self._exec_config_str(k, v)

        for subc in subcs:
            self._load_flag(subc)

        # if self.extra_args:
        #    sub_parser = KeyValueConfigLoader(log=self.log)
        #    sub_parser.load_config(self.extra_args)
        #    self.config.merge(sub_parser.config)
        #    self.extra_args = sub_parser.extra_args


class FactoryTool(Tool):
    """
    Proposal for complete relacement of Tool
    """
    factories = Dict(dict())

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_classes = copy(self.classes)
        self.init_aliases = copy(self.aliases)

    def run(self, argv=None):
        """Run the tool. This automatically calls `initialize()`,
        `start()` and `finish()`
        """
        try:
            if self.factories:
                self.factory_initialize(argv)
            self.initialize(argv)
            self.log.info("Starting: {}".format(self.name))
            self.log.debug("CONFIG: {}".format(self.config))
            self.start()
            self.finish()
        except ValueError as err:
            self.log.error('{}'.format(err))
        except RuntimeError as err:
            self.log.error('Caught unexpected exception: {}'.format(err))

    def factory_initialize(self, argv=None):
        self.classes = copy(self.init_classes)
        self.aliases = copy(self.init_aliases)
        for factory in self.factories.values():
            if factory not in self.classes:
                self.classes.append(factory)
        self.parse_known_command_line(argv)
        if self.config_file != '':
            self.log.debug("Loading config from '{}'".format(self.config_file))
            self.load_config_file(self.config_file)
        for key, val in self.factories.items():
            product = factory(config=self.config).init_product()
            self.classes.append(product)
            for trait in product.class_trait_names(config=True):
                self.aliases[trait] = product.__name__ + '.' + trait


    @catch_config_error
    def parse_known_command_line(self, argv=None):
        """
        Parse the known command line arguments.
        Same as Tool.parse_command_line, except calls the CustomLoader,
        and does not print help.
        """
        argv = sys.argv[1:] if argv is None else argv
        self.argv = [py3compat.cast_unicode(arg) for arg in argv]

        if argv and argv[0] == 'help':
            # turn `ipython help notebook` into `ipython notebook -h`
            argv = argv[1:] + ['-h']

        if self.subcommands and len(argv) > 0:
            # we have subcommands, and one may have been specified
            subc, subargv = argv[0], argv[1:]
            if re.match(r'^\w(\-?\w)*$', subc) and subc in self.subcommands:
                # it's a subcommand, and *not* a flag or class parameter
                return self.initialize_subcommand(subc, subargv)

        # Arguments after a '--' argument are for the script IPython may be
        # about to run, not IPython iteslf. For arguments parsed here (help and
        # version), we want to only search the arguments up to the first
        # occurrence of '--', which we're calling interpreted_argv.
        try:
            interpreted_argv = argv[:argv.index('--')]
        except ValueError:
            interpreted_argv = argv

        # Remove help from argv, this will be parsed in parse_command_line
        # after factories are set
        unwanted = ['-h', '--help-all', '--help', '--version']
        argv = [string for string in argv if string not in unwanted]

        # flatten flags&aliases, so cl-args get appropriate priority:
        flags, aliases = self.flatten_flags()
        loader = CustomLoader(argv=argv, aliases=aliases,
                              flags=flags, log=self.log)
        self.cli_config = deepcopy(loader.load_config())
        self.update_config(self.cli_config)
        # store unparsed args in extra_args
        self.extra_args = loader.extra_args