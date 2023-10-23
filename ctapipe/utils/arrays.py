"""
Some utility functions to work with numpy (rec) arrays
"""
import numpy as np
from numpy.lib.recfunctions import repack_fields


def recarray_drop_columns(array, columns):
    """
    Remove columns from rec array
    """
    to_use = [col for col in array.dtype.names if col not in columns]
    subset = array.view(subset_dtype(array.dtype, to_use))
    return repack_fields(subset)


def subset_dtype(dtype, names):
    """
    Create a new dtype only containing a subset of the columns
    """
    formats = [dtype.fields[name][0] for name in names]
    offsets = [dtype.fields[name][1] for name in names]
    itemsize = dtype.itemsize
    return np.dtype(
        dict(names=names, formats=formats, offsets=offsets, itemsize=itemsize)
    )
