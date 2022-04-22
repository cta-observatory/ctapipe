import numpy as np
from numpy.lib.recfunctions import repack_fields


def recarray_drop_columns(a, columns):
    '''
    Remove columns from rec array
    '''
    to_use = [col for col in a.dtype.names if col not in columns]
    subset = a.view(subset_dtype(a.dtype, to_use))
    return repack_fields(subset)


def subset_dtype(dtype, names):
    '''
    Create a new dtype only containing a subset of the columns
    '''
    formats = [dtype.fields[name][0] for name in names]
    offsets = [dtype.fields[name][1] for name in names]
    itemsize = dtype.itemsize
    return np.dtype(dict(
        names=names,
        formats=formats,
        offsets=offsets,
        itemsize=itemsize
    ))
