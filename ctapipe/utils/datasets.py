# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
from pathlib import Path

__all__ = ['CTAPipeDatasetsNotFoundError',
           'get_ctapipe_extra_path',
           'get_datasets_path',
           ]


class CTAPipeDatasetsNotFoundError(RuntimeError):
    """ctapipe datasets not found error.
    """
    def __init__(self, path):
        message = "Does not exist: '{}'".format(path)
        super(RuntimeError, self).__init__(message)
    

def get_ctapipe_extra_path(environ_variable_name='CTAPIPE_EXTRA_DIR'):
    """Get path to `ctapipe-extra`.

    First try shell environment variable.
    Then try git submodule in the right location.
    """
    try:
        return Path(os.environ[environ_variable_name])
    except KeyError:
        pass

    import ctapipe
    path = Path(ctapipe.__file__).parent.parent.joinpath('ctapipe-extra')
    if path.exists():
        return path

    raise CTAPipeDatasetsNotFoundError(path)


def get_datasets_path(file_path):
    path = Path(get_ctapipe_extra_path(), 'datasets', file_path)
    return path.as_posix()


def get_path(file_path):
    return get_datasets_path(file_path)
