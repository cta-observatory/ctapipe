"""
Module for python version compatibility
"""
import sys

import numpy as np
from packaging.version import Version

__all__ = [
    "StrEnum",
]


if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        """Compatibility backfill of StrEnum for python < 3.11"""


# in numpy 1.x, copy=False allows copying if it cannot be avoided
# in numpy 2.0, copy=False raises an error when the copy cannot be avoided
# copy=None is a new option in numpy 2.0 for the previous behavior of copy=False
COPY_IF_NEEDED = None
if Version(np.__version__) < Version("2.0.0.dev"):
    COPY_IF_NEEDED = False
