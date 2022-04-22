import numpy as np


def test_drop_columns():
    from ctapipe.utils.arrays import recarray_drop_columns
    array = np.empty(
        1000,
        dtype=[
            ('x', float),
            ('N', int),
            ('valid', bool),
            ('tel_id', np.uint16)
        ]
    )
    assert array.dtype.itemsize == 19

    dropped = recarray_drop_columns(array, columns=['N', 'tel_id'])
    assert dropped.dtype.names == ('x', 'valid')
    assert dropped.dtype.itemsize == 9
