# Licensed under a 3-clause BSD style license - see LICENSE.rst
""" print information about ctapipe and its command-line tools. """
import sys
import logging
import importlib
from .utils import get_parser

__all__ = ['info']


def main(args=None):
    parser = get_parser(info)
    parser.add_argument('--version', action='store_true',
                        help='Print version number')
    parser.add_argument('--tools', action='store_true',
                        help='Print available command line tools')
    parser.add_argument('--dependencies', action='store_true',
                        help='Print available versions of dependencies')
    args = parser.parse_args(args)

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(1)

    info(**vars(args))


def info(version=False, tools=False, dependencies=False):
    """Print various info to the console.

    TODO: explain.
    """
    logging.basicConfig(level=logging.DEBUG,
                        format='%(levelname)s - %(message)s')

    if version:
        _info_version()

    if tools:
        _info_tools()

    if dependencies:
        _info_dependencies()


def _info_version():
    """Print version info."""
    import ctapipe
    print('\n*** ctapipe version info ***\n')
    print('version: {0}'.format(ctapipe.__version__))
    #print('release: {0}'.format(version.release))
    #print('githash: {0}'.format(version.githash))
    print('')


def _info_tools():
    """Print info about command line tools."""
    print('\n*** ctapipe tools ***\n')
    print('the following can be executed by typing ctapipe-<toolname>:')
    print('')

    # TODO: how to get a one-line description or
    # full help text from the docstring or ArgumentParser?
    # This is the function names, we want the command-line names
    # that are defined in setup.py !???
    from ctapipe.tools.utils import get_all_descriptions
    from textwrap import TextWrapper
    wrapper = TextWrapper(width=80,
                          subsequent_indent=" "*35 )
                          
    scripts = get_all_descriptions()
    for name, desc in sorted(scripts.items()):
        text ="{:<30s}  - {}".format(name, desc) 
        print(wrapper.fill(text))
        print('')
    print('')



def _info_dependencies():
    """Print info about dependencies."""
    print('\n*** ctapipe dependencies ***\n')
    from ctapipe.conftest import PYTEST_HEADER_MODULES

    for label, name in PYTEST_HEADER_MODULES.items():
        try:
            module = importlib.import_module(name)
            version = module.__version__
        except ImportError:
            version = 'not available'

        print('{:>20s} -- {}'.format(label, version))
