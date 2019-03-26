# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utils to create scripts and command-line tools"""
import argparse
import importlib
import re
from collections import OrderedDict

__all__ = ['ArgparseFormatter',
           'get_parser',
           'get_installed_tools',
           'get_all_descriptions',
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


def get_installed_tools():
    """Get list of installed scripts via ``pkg-resources``.

    See http://peak.telecommunity.com/DevCenter/PkgResources#convenience-api

    TODO: not sure if this will be useful ... maybe to check if the list
    of installed packages matches the available scripts somehow?
    """
    from pkg_resources import get_entry_map
    console_tools = get_entry_map('ctapipe')['console_scripts']
    return console_tools


def get_all_descriptions():

    tools = get_installed_tools()

    descriptions = OrderedDict()
    for name, info in tools.items():
        module = importlib.import_module(info.module_name)
        if hasattr(module, '__doc__') and module.__doc__ is not None:
            try:
                descrip = module.__doc__
                descrip.replace("\n", "")
                descriptions[name] = descrip
            except Exception as err:
                descriptions[name] = f"[Couldn't parse docstring: {err}]"
        else:
            descriptions[name] = "[no documentation. Please add a docstring]"

    return descriptions

