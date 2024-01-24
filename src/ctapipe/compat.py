"""
Module for python version compatibility
"""
import sys

__all__ = [
    "StrEnum",
]


if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        """Compatibility backfill of StrEnum for python < 3.11"""
