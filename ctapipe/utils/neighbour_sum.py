"""
Conveniance module to handle the calling of functions within
neighbour_sum_c.cc through ctypes.
"""

__all__ = []

import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer
import os

try:
    lib = np.ctypeslib.load_library("neighbour_sum_c", os.path.dirname(__file__))
    get_sum_array = lib.get_sum_array
    get_sum_array.restype = None
    get_sum_array.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                              ndpointer(ctypes.c_uint16, flags="C_CONTIGUOUS"),
                              ctypes.c_size_t,
                              ctypes.c_int]
except OSError:
    pass
