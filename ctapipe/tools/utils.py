# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utils to create scripts and command-line tools"""
import argparse
import importlib
import sys
from collections import OrderedDict

if sys.version_info < (3, 10):
    from importlib_metadata import distribution
else:
    from importlib.metadata import distribution

__all__ = [
    "ArgparseFormatter",
    "get_parser",
    "get_installed_tools",
    "get_all_descriptions",
]


class ArgparseFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter
):
    """ArgumentParser formatter_class argument."""

    pass


def get_parser(function=None, description="N/A"):
    """Make an ArgumentParser how we like it."""
    if function:
        description = function.__doc__
    parser = argparse.ArgumentParser(
        description=description, formatter_class=ArgparseFormatter
    )
    return parser


def get_installed_tools():
    """Get list of installed scripts via ``pkg-resources``.

    See https://setuptools.pypa.io/en/latest/pkg_resources.html#convenience-api

    TODO: not sure if this will be useful ... maybe to check if the list
    of installed packages matches the available scripts somehow?
    """
    console_tools = {
        ep.name: ep.value
        for ep in distribution("ctapipe").entry_points
        if ep.group == "console_scripts"
    }
    return console_tools


def get_all_descriptions():
    tools = get_installed_tools()

    descriptions = OrderedDict()
    for name, value in tools.items():
        module_name, attr = value.split(":")
        module = importlib.import_module(module_name)
        if hasattr(module, "__doc__") and module.__doc__ is not None:
            try:
                descrip = module.__doc__
                descrip.replace("\n", "")
                descriptions[name] = descrip
            except Exception as err:
                descriptions[name] = f"[Couldn't parse docstring: {err}]"
        else:
            descriptions[name] = "[no documentation. Please add a docstring]"

    return descriptions
