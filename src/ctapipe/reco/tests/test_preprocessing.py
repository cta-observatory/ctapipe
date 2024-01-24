import astropy.units as u
import numpy as np
from astropy.table import Table
from numpy.testing import assert_array_equal


def test_table_to_float32():
    from ctapipe.reco.preprocessing import table_to_float

    t = Table({"a": [1.0, 1e50, np.inf, -np.inf, np.nan], "b": [1, 2, 3, 4, 5]})

    fmax = np.finfo(np.float32).max
    fmin = np.finfo(np.float32).min
    expected = np.array(
        [[1.0, fmax, fmax, fmin, np.nan], [1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32
    ).T

    array = table_to_float(t)
    assert array.dtype == np.float32
    assert_array_equal(array, expected)


def test_table_to_float32_units():
    from ctapipe.reco.preprocessing import table_to_float

    t = Table(
        {"a": [1.0, 1e50, np.inf, -np.inf, np.nan] * u.m, "b": [1, 2, 3, 4, 5] * u.deg}
    )

    fmax = np.finfo(np.float32).max
    fmin = np.finfo(np.float32).min
    expected = np.array(
        [[1.0, fmax, fmax, fmin, np.nan], [1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32
    ).T

    array = table_to_float(t)
    assert array.dtype == np.float32
    assert_array_equal(array, expected)


def test_table_to_float64():
    from ctapipe.reco.preprocessing import table_to_float

    t = Table({"a": [1.0, 1e50, np.inf, -np.inf, np.nan], "b": [1, 2, 3, 4, 5]})

    fmax = np.finfo(np.float64).max
    fmin = np.finfo(np.float64).min
    expected = np.array(
        [[1.0, 1e50, fmax, fmin, np.nan], [1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float64
    ).T

    array = table_to_float(t, dtype=np.float64)
    assert_array_equal(array, expected)
    assert array.dtype == np.float64
    assert_array_equal(array, expected)


def test_check_valid_rows():
    from ctapipe.reco.preprocessing import check_valid_rows

    t = Table({"a": [1.0, 2, np.inf, np.nan, np.nan], "b": [1, np.inf, 3, 4, 5]})

    valid = check_valid_rows(t, warn=False)
    assert_array_equal(valid, [True, True, True, False, False])
