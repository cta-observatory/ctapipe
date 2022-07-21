"""
ctapipe - CTA Python pipeline experimental version

Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
from .core.plugins import IMPORTED_PLUGINS, detect_and_import_plugins
from .version import __version__

__all__ = ["__version__", "IMPORTED_PLUGINS"]


detect_and_import_plugins()
