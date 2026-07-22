"""ctapipe plugin system"""

import logging
from functools import lru_cache
from importlib.metadata import entry_points

log = logging.getLogger(__name__)
installed_entry_points = entry_points()


def detect_and_import_plugins(group):
    """detect and import plugins with given prefix,"""
    modules = set()
    for entry_point in installed_entry_points.select(group=group):
        log.debug("Loading %s plugin: %s", group, entry_point.value)
        try:
            plugin = entry_point.load()
            if plugin.__module__.split(".")[0] != "ctapipe":
                modules.add(plugin.__module__)
            log.debug("Entrypoint provides: %s", plugin)
        except Exception:
            log.exception("Error loading %s plugin: '%s'", group, entry_point.value)

    return tuple(modules)


def detect_and_import_io_plugins():
    """Import io plugins (providing event sources)

    These plugins are meant to provide `~ctapipe.io.EventSource` implementations
    """
    return detect_and_import_plugins(group="ctapipe_io")


def detect_and_import_reco_plugins():
    """Import reco plugins

    These plugins are meant to provide `~ctapipe.reco.Reconstructor` implementations
    """
    return detect_and_import_plugins(group="ctapipe_reco")


@lru_cache(maxsize=None)
def resolve_entry_point(group, name):
    """Load a single entry point by group and exact name.

    Parameters
    ----------
    group : str
        entry-point group to look up, e.g. "ctapipe_monitoring_containers"
    name : str
        entry-point name within `group`, e.g. "drive"

    Returns
    -------
    object or None
        the loaded object, or None if no plugin registers `name` under
        `group`.

    Raises
    ------
    RuntimeError
        if more than one plugin registers the same name within `group`.
    """
    matches = list(installed_entry_points.select(group=group, name=name))

    if not matches:
        return None

    if len(matches) > 1:
        candidates = [m.value for m in matches]
        raise RuntimeError(
            f"Multiple plugins registered {name!r} under entry-point "
            f"group {group!r}: {candidates}."
        )

    entry_point = matches[0]
    log.debug("Loading %s plugin: %s", group, entry_point.value)
    return entry_point.load()


def lazy_entry_point(group, name, qualname=None):
    """Build a class usable as a Container Field's `default_factory` that
    resolves and instantiates a plugin-registered class the first time it
    is called, rather than when this function is called.

    Resolves to None at call time if no plugin registers `name` under
    `group`.

    Parameters
    ----------
    group : str
        entry-point group to look up
    name : str
        entry-point name within `group`
    qualname : str, optional
        name to report in reprs/docs for the returned class; defaults to
        `name`

    Returns
    -------
    type
        a class whose instantiation resolves and instantiates the
        plugin-registered class, or returns None if unregistered.
    """

    class _LazyEntryPointContainer:
        def __new__(cls):
            target_cls = resolve_entry_point(group, name)
            if target_cls is None:
                return None
            return target_cls()

    _LazyEntryPointContainer.__name__ = qualname or name
    _LazyEntryPointContainer.__qualname__ = _LazyEntryPointContainer.__name__
    return _LazyEntryPointContainer
