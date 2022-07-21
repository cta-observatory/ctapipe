""" Functions for dealing with IO plugins """
import importlib
import logging
import pkgutil

__all__ = [
    "detect_and_import_plugins",
    "IMPORTED_PLUGINS",
]


log = logging.getLogger(__name__)

IMPORTED_PLUGINS = {}


def detect_and_import_plugins(prefix="ctapipe_"):
    """detect and import  plugin modules with given prefix,"""

    for _, name, _ in pkgutil.iter_modules():
        if not name.startswith(prefix):
            continue

        try:
            IMPORTED_PLUGINS[name] = importlib.import_module(name)
            log.info("Imported plugin %s", name)
        except Exception as e:
            log.error("Failed to import pluging %s: %s", name, e)
