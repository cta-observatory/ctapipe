# Licensed under a 3-clause BSD style license - see LICENSE.rst
""" print information about ctapipe and its command-line tools. """
import importlib
import logging
import os
import sys

from .utils import get_parser
from ..core import Provenance, get_module_version
from ..utils import datasets

__all__ = ['info']

# TODO: this list should be global (or generated at install time)
_dependencies = sorted([
    'astropy', 'matplotlib',
    'numpy', 'traitlets',
    'sklearn', 'scipy', 'numba',
    'pytest', 'iminuit', 'tables',
    'eventio',
])

_optional_dependencies = sorted([
    'ctapipe_resources',
    'pytest',
    'graphviz',
    'pyhessio',
    'targetio',
    'matplotlib'
])


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
    parser.add_argument('--system', action='store_true',
                        help='Print system info')
    parser.add_argument('--all', dest='show_all', action='store_true',
                        help='show all info')
    args = parser.parse_args(args)

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(1)

    info(**vars(args))


def info(version=False, tools=False, dependencies=False,
         resources=False, system=False, show_all=False):
    """Print various info to the console.

    TODO: explain.
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s - %(message)s')

    if version or show_all:
        _info_version()

    if tools or show_all:
        _info_tools()

    if dependencies or show_all:
        _info_dependencies()

    if resources or show_all:
        _info_resources()

    if system or show_all:
        _info_system()


def _info_version():
    """Print version info."""
    import ctapipe
    print('\n*** ctapipe version info ***\n')
    print(f'version: {ctapipe.__version__}')
    # print('release: {0}'.format(version.release))
    # print('githash: {0}'.format(version.githash))
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
                          subsequent_indent=" " * 35)

    scripts = get_all_descriptions()
    for name, desc in sorted(scripts.items()):
        text = f"{name:<30s}  - {desc}"
        print(wrapper.fill(text))
        print('')
    print('')


def _info_dependencies():
    """Print info about dependencies."""
    print('\n*** ctapipe core dependencies ***\n')

    for name in _dependencies:
        version = get_module_version(name)
        print(f'{name:>20s} -- {version}')

    print('\n*** ctapipe optional dependencies ***\n')

    for name in _optional_dependencies:
        try:
            module = importlib.import_module(name)
            version = module.__version__
        except ImportError:
            version = 'not installed'
        except AttributeError:
            version = "installed, but __version__ doesn't exist"

        print(f'{name:>20s} -- {version}')


def _info_resources():
    """ display all known resources """

    print('\n*** ctapipe resources ***\n')
    print("CTAPIPE_SVC_PATH: (directories where resources are searched)")
    if os.getenv('CTAPIPE_SVC_PATH') is not None:
        for directory in datasets.get_searchpath_dirs():
            print(f"\t * {directory}")
    else:
        print("\t no path is set")
    print("")

    all_resources = sorted(datasets.find_all_matching_datasets(r'\w.*'))
    locations = [os.path.dirname(datasets.get_dataset_path(name))
                 for name in all_resources]
    home = os.path.expanduser("~")
    resource_dir = os.path.dirname(datasets.get_dataset_path(
        "HESS-I.camgeom.fits.gz"))

    fmt = "{name:<30.30s} : {loc:<30.30s}"
    print(fmt.format(name="RESOURCE NAME", loc="LOCATION"))
    print("-" * 70)
    for name, loc in zip(all_resources, locations):
        if name.endswith(".py") or name.startswith("_"):
            continue
        loc = loc.replace(resource_dir, "[ctapipe_resources]")
        loc = loc.replace(home, "~")
        print(fmt.format(name=name, loc=loc))


def _info_system():
    # collect system info using the ctapipe provenance system :

    print('\n*** ctapipe system environment ***\n')

    prov = Provenance()
    system_prov = prov.current_activity.provenance['system']

    for section in ['platform', 'python']:

        print('\n====== ', section, " ======== \n")
        sysinfo = system_prov[section]

        for name, val in sysinfo.items():
            print("{:>20.20s} -- {:<60.60s}".format(name, str(val)))


if __name__ == '__main__':
    main()
