# Licensed under a 3-clause BSD style license - see LICENSE.rst
""" print information about ctapipe and its command-line tools. """
import sys
import logging
import importlib
from .utils import get_parser
from ctapipe.utils import datasets
import os
import ctapipe_resources

__all__ = ['info']


# TODO: this list should be global (or generated at install time)
_dependencies = sorted(['astropy', 'matplotlib',
                        'numpy', 'traitlets',
                        'sklearn','scipy',
                        'pytest', 'ctapipe_resources'])

_optional_dependencies = sorted(['pytest','graphviz','pyzmq','iminuit',
                                 'fitsio','pyhessio','targetio'])


def main(args=None):
    parser = get_parser(info)
    parser.add_argument('--version', action='store_true',
                        help='Print version number')
    parser.add_argument('--tools', action='store_true',
                        help='Print available command line tools')
    parser.add_argument('--dependencies', action='store_true',
                        help='Print available versions of dependencies')
    parser.add_argument('--resources', action='store_true',
                        help='Print available versions of dependencies')
    parser.add_argument('--all', action='store_true',
                        help='show all info')
    args = parser.parse_args(args)

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(1)

    info(**vars(args))


def info(version=False, tools=False, dependencies=False,
         resources=False, all=False):
    """Print various info to the console.

    TODO: explain.
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s - %(message)s')

    if version or all:
        _info_version()

    if tools or all:
        _info_tools()

    if dependencies or all:
        _info_dependencies()

    if resources or all:
        _info_resources()

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
    print('\n*** ctapipe core dependencies ***\n')

    for name in _dependencies:
        try:
            module = importlib.import_module(name)
            version = module.__version__
        except ImportError:
            version = 'not installed'

        print('{:>20s} -- {}'.format(name, version))

    print('\n*** ctapipe optional dependencies ***\n')

    for name in _optional_dependencies:
        try:
            module = importlib.import_module(name)
            version = module.__version__
        except ImportError:
            version = 'not installed'
        except AttributeError:
            version = "installed, but __version__ doesn't exist"

        print('{:>20s} -- {}'.format(name, version))


def _info_resources():
    """ display all known resources """

    print('\n*** ctapipe resources ***\n')

    print("ctapipe_resources version: {}".format(ctapipe_resources.__version__))

    print("CTAPIPE_SVC_PATH: (directories where resources are searched)")
    if os.getenv('CTAPIPE_SVC_PATH') is not None:
        for dir in datasets.get_searchpath_dirs():
            print("\t * {}".format(dir))
    else:
        print("\t no path is set")
    print("")

    all_resources = sorted(datasets.find_all_matching_datasets("\w.*"))
    locations = [os.path.dirname(datasets.get_dataset(name))
                 for name in all_resources]
    home = os.path.expanduser("~")
    resource_dir = os.path.dirname(datasets.get_dataset(
        "HESS-I.camgeom.fits.gz"))

    fmt = "{name:<30.30s} : {loc:<30.30s}"
    print(fmt.format(name="RESOURCE NAME", loc="LOCATION"))
    print("-"*70)
    for name, loc  in zip(all_resources, locations):
        if name.endswith(".py") or name.startswith("_"):
            continue
        loc = loc.replace(resource_dir, "[ctapipe_resources]")
        loc = loc.replace(home, "~")
        print(fmt.format(name=name, loc=loc))

