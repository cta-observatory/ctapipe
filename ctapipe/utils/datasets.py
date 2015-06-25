# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os


def get_datasets_path(environ_variable_name='CTAPIPE_EXTRA_DIR'):
    """Get path to test and example datasets.

    First try shell environment variable.
    Then try git submodule in the right location.
    """
    try:
        return os.environ[environ_variable_name]
    except KeyError:
        pass

    import ctapipe
    path = os.path.join(os.path.dirname(ctapipe.__file__), os.pardir, 'ctapipe-extra', 'datasets')
    print(path)
    if os.path.exists(path):
        return path

    return None
