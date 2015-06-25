# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
from pathlib import Path

__all__ = ['CTAPipeDatasetsNotFoundError',
           'get_datasets_path',
           'get_file',
           ]


class CTAPipeDatasetsNotFoundError(Exception):
    """ctapipe datasets not found error.
    """


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
    path = Path(ctapipe.__file__).parent.joinpath('ctapipe-extra')
    if path.exists():
        return path

    raise CTAPipeDatasetsNotFoundError


def get_path(file_path):
    path = Path(get_ctapipe_extra_path(), 'datasets', file_path)
    return path.as_posix()
