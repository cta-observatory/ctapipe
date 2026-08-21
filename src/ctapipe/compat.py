"""
Module for python version compatibility
"""

import numpy as np
from packaging.version import Version

__all__ = [
    "COPY_IF_NEEDED",
    "ECSV_FMT",
    "trapz_func",
]


# in numpy 1.x, copy=False allows copying if it cannot be avoided
# in numpy 2.0, copy=False raises an error when the copy cannot be avoided
# copy=None is a new option in numpy 2.0 for the previous behavior of copy=False
COPY_IF_NEEDED = None
if Version(np.__version__) < Version("2.0.0.dev"):
    COPY_IF_NEEDED = False

# in numpy 1.x, use trapz; in numpy 2.0+, trapz is deprecated in favor of trapezoid
try:
    trapz_func = np.trapezoid
except AttributeError:
    trapz_func = np.trapz


# TODO: astropy introduced a new "ecsv" parser in 7.2, but we
# currently cannot use it due to it not parsing empty tables:
# https://github.com/astropy/astropy/issues/19895
# if Version(astropy.__version__) < Version("7.2.0.dev0"):
#     ECSV_FMT = "ascii.ecsv"
# else:
#     ECSV_FMT = "ecsv"
ECSV_FMT = "ascii.ecsv"
