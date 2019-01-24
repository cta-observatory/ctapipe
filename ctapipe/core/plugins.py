import importlib
import pkgutil


def detect_and_import_plugins(prefix):
    ''' detect and import  plugin modules with given prefix, '''
    return {
        name: importlib.import_module(name)
        for finder, name, ispkg
        in pkgutil.iter_modules()
        if name.startswith(prefix)
    }


def detect_and_import_io_plugins():
    return detect_and_import_plugins(prefix='ctapipe_io_')
