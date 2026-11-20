"""
Environment variables to configure ctapipe.
"""

import os

__all__ = [
    "CTAPIPE_DISABLE_NUMBA_CACHE",
]


def env_bool(key, default=False):
    """
    Parse a boolean environment variable.

    ``"1"``, ``"true"`` and ``"on"`` are treated as truthy, case-insensitive.

    If the variable is not set, ``default`` is returned.

    Parameters
    ----------
    key : str
        Name of the environment variable to interpret.
    default : bool, optional
        Value to return when the variable is unset.
    """
    val = os.getenv(key)
    if val is None:
        return default

    return val.lower() in {"1", "true", "on"}


#: Boolean flag. Set this variable to a truthy value disable numba caching.
CTAPIPE_DISABLE_NUMBA_CACHE = env_bool("CTAPIPE_DISABLE_NUMBA_CACHE")
