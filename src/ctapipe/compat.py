"""
Module for python version compatibility
"""

import numpy as np
from packaging.version import Version

__all__ = [
    "COPY_IF_NEEDED",
]


# in numpy 1.x, copy=False allows copying if it cannot be avoided
# in numpy 2.0, copy=False raises an error when the copy cannot be avoided
# copy=None is a new option in numpy 2.0 for the previous behavior of copy=False
COPY_IF_NEEDED = None
if Version(np.__version__) < Version("2.0.0.dev"):
    COPY_IF_NEEDED = False
