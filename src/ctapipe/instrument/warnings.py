"""Warnings related to the instrument"""
import warnings

__all__ = ["FromNameWarning", "warn_from_name"]


class FromNameWarning(UserWarning):
    """Warning raised when using from_name"""


def warn_from_name():
    """Warning raised when accessing .from_name methods"""
    msg = (
        ".from_name uses pre-defined data that is"
        " likely different from the data being analyzed."
        " Access instrument information via the SubarrayDescription instead."
    )
    warnings.warn(msg, FromNameWarning, stacklevel=2)
