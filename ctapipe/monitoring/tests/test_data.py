import pytest
import numpy as np
from astropy.time import Time
import astropy.units as u


@pytest.fixture
def mon_int():
    from ctapipe.monitoring import MonitoringData

    indices = np.arange(5)
    values = 5 * indices + 2

    return MonitoringData(zip(indices, values))


def test_int_before(mon_int):
    assert mon_int.before(-0.5) == (None, None)
    assert mon_int.before(0) == (0, 2)
    assert mon_int.before(0, inclusive=False) == (None, None)

    assert mon_int.before(1) == (1, 7)
    assert mon_int.before(1, inclusive=False) == (0, 2)


def test_int_after(mon_int):
    assert mon_int.after(0) == (0, 2)
    assert mon_int.after(0, inclusive=False) == (1, 7)

    assert mon_int.after(4) == (4, 22)
    assert mon_int.after(4, inclusive=False) == (None, None)


def test_int_closest(mon_int):
    assert mon_int.closest(-0.5) == (0, 2)
    assert mon_int.closest(0.0) == (0, 2)
    assert mon_int.closest(0.4) == (0, 2)

    # if distance is the same, currently the "before" value is returned
    assert mon_int.closest(0.5) == (0, 2)

    assert mon_int.closest(1.6) == (2, 12)
    assert mon_int.closest(2.0) == (2, 12)
    assert mon_int.closest(2.4) == (2, 12)
    assert mon_int.closest(2.5) == (2, 12)


def test_int_interpolate(mon_int):
    # extrapolation to below
    assert mon_int.interpolate_linear(-0.5) == -0.5

    # interpolation on a support point
    assert mon_int.interpolate_linear(0) == 2

    # normal interpolation
    assert mon_int.interpolate_linear(0.5) == 4.5

    # extrapolation to above
    assert mon_int.interpolate_linear(5) == 27


@pytest.fixture
def mon_time():
    from ctapipe.monitoring import MonitoringData

    indices = [
        Time('2020-01-01T12:00:00'),
        Time('2020-01-01T13:00:00'),
        Time('2020-01-01T14:00:00'),
        Time('2020-01-01T15:00:00'),
        Time('2020-01-01T16:00:00'),
    ]
    dt = np.arange(5)
    values = 5 * dt + 2

    return MonitoringData(zip(indices, values))


def test_time_before(mon_time):
    t11 = Time('2020-01-01T11:00:00')
    t12 = Time('2020-01-01T12:00:00')
    t13 = Time('2020-01-01T13:00:00')

    assert mon_time.before(t11) == (None, None)
    assert mon_time.before(t12) == (t12, 2)
    assert mon_time.before(t12, inclusive=False) == (None, None)

    assert mon_time.before(Time('2020-01-01T13:30:00')) == (t13, 7)
    assert mon_time.before(t13, inclusive=False) == (t12, 2)


def test_time_after(mon_time):
    t14 = Time('2020-01-01T14:00:00')
    t15 = Time('2020-01-01T15:00:00')
    t16 = Time('2020-01-01T16:00:00')
    t17 = Time('2020-01-01T17:00:00')
    assert mon_time.after(t17) == (None, None)
    assert mon_time.after(t16) == (t16, 22)
    assert mon_time.after(t16, inclusive=False) == (None, None)

    assert mon_time.after(Time('2020-01-01T13:30:00')) == (t14, 12)
    assert mon_time.after(t14, inclusive=False) == (t15, 17)


def test_time_closest(mon_time):
    t11_40 = Time('2020-01-01T11:40:00')
    t12 = Time('2020-01-01T12:00:00')
    t12_20 = Time('2020-01-01T12:20:00')
    t12_40 = Time('2020-01-01T12:40:00')
    t13 = Time('2020-01-01T13:00:00')

    assert mon_time.closest(t11_40) == (t12, 2)
    assert mon_time.closest(t12) == (t12, 2)
    assert mon_time.closest(t12_20) == (t12, 2)
    assert mon_time.closest(t12_40) == (t13, 7)


def test_time_interpolate(mon_time):
    t11_30 = Time('2020-01-01T11:30:00')
    t12 = Time('2020-01-01T12:00:00')
    t12_30 = Time('2020-01-01T12:30:00')
    t17 = Time('2020-01-01T17:00:00')

    # extrapolation to below
    assert np.isclose(mon_time.interpolate_linear(t11_30), -0.5)

    # interpolation on a support point
    assert np.isclose(mon_time.interpolate_linear(t12), 2)

    # normal interpolation
    assert np.isclose(mon_time.interpolate_linear(t12_30), 4.5)

    # extrapolation to above
    assert np.isclose(mon_time.interpolate_linear(t17), 27)
