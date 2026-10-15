"""Tests for the array utils"""

import numpy as np
import pytest


def test_drop_columns():
    """Test drop columns"""
    from ctapipe.utils.arrays import recarray_drop_columns

    array = np.empty(
        1000, dtype=[("x", float), ("N", int), ("valid", bool), ("tel_id", np.uint16)]
    )
    assert array.dtype.itemsize == 19

    dropped = recarray_drop_columns(array, columns=["N", "tel_id"])
    assert dropped.dtype.names == ("x", "valid")
    assert dropped.dtype.itemsize == 9


@pytest.mark.parametrize("n", [1, 2, 10, 1855])
def test_cached_arange(n):
    from ctapipe.utils.arrays import cached_arange

    n1 = cached_arange(n)
    n2 = cached_arange(n)
    assert n1 is n2
    np.testing.assert_array_equal(n1, np.arange(n))

    with pytest.raises(ValueError):
        n1[2] = -1
