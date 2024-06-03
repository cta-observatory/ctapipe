"""ctapipe plugin system"""
import logging
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
