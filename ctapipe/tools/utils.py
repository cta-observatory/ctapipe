# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utils to create scripts and command-line tools"""
import argparse
from collections import OrderedDict
import importlib
import os
import glob
import re

__all__ = ['ArgparseFormatter',
           'get_parser',
           'get_installed_scripts',
           'get_all_main_functions',
           ]


class ArgparseFormatter(argparse.ArgumentDefaultsHelpFormatter,
                        argparse.RawTextHelpFormatter):
    """ArgumentParser formatter_class argument.
    """
    pass


def get_parser(function=None, description='N/A'):
    """Make an ArgumentParser how we like it.
    """
    if function:
        description = function.__doc__.split('\n')[0]
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=ArgparseFormatter)
    return parser


def get_installed_scripts():
    """Get list of installed scripts via ``pkg-resources``.

    See http://peak.telecommunity.com/DevCenter/PkgResources#convenience-api

    TODO: not sure if this will be useful ... maybe to check if the list
    of installed packages matches the available scripts somehow?
    """
    from pkg_resources import get_entry_map
    console_scripts = get_entry_map('ctapipe')['console_scripts']
    return console_scripts


def get_all_main_functions():
    """Get a dict with all scripts (used for testing).
    """
    # Could this work?
    # http://stackoverflow.com/questions/1707709/list-all-the-modules-that-are-part-of-a-python-package
    # import pkgutil
    # pkgutil.iter_modules(path=None, prefix='')

    path = os.path.join(os.path.dirname(__file__), '../tools')
    names = glob.glob1(path, '*.py')
    names = [_.replace('.py', '') for _ in names]
    for name in ['__init__']:
        names.remove(name)

    out = OrderedDict()
    for name in names:
        module = importlib.import_module('ctapipe.tools.{}'.format(name))
        if hasattr(module, 'main'):
            out[name] = module.main

    return out


def get_all_descriptions():

    mains = get_all_main_functions()

    descriptions = OrderedDict()
    for name in mains.keys():
        module = importlib.import_module('ctapipe.tools.{}'.format(name))
        if hasattr(module, '__doc__'):
            try:
                descrip = re.match(r'(?:[^.:;]+[.:;]){1}',
                                   module.__doc__).group()
                descrip.replace("\n", "")
                descriptions[name] = descrip
            finally:
                descriptions[name] = "[no documentation]"

    return descriptions
    
