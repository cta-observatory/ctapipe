import astropy.units as u
import numpy as np
import pytest
import tables
from astropy.table import Table
from astropy.time import Time


def test_simple():
    """Test pointing interpolation"""
    from ctapipe.io.pointing import PointingInterpolator

    t0 = Time("2022-01-01T00:00:00")

    table = Table(
        {
            "time": t0 + np.arange(0.0, 10.1, 2.0) * u.s,
            "azimuth": np.linspace(0.0, 10.0, 6) * u.deg,
            "altitude": np.linspace(70.0, 60.0, 6) * u.deg,
        },
    )

    interpolator = PointingInterpolator()
    interpolator.add_table(1, table)

    alt, az = interpolator(tel_id=1, time=t0 + 1 * u.s)
    assert u.isclose(alt, 69 * u.deg)
    assert u.isclose(az, 1 * u.deg)

    with pytest.raises(KeyError):
        interpolator(tel_id=2, time=t0 + 1 * u.s)


def test_azimuth_switchover():
    """Test pointing interpolation"""
    from ctapipe.io.pointing import PointingInterpolator

    t0 = Time("2022-01-01T00:00:00")

    table = Table(
        {
            "time": t0 + [0, 1, 2] * u.s,
            "azimuth": [359, 1, 3] * u.deg,
            "altitude": [60, 61, 62] * u.deg,
        },
    )

    interpolator = PointingInterpolator()
    interpolator.add_table(1, table)

    alt, az = interpolator(tel_id=1, time=t0 + 0.5 * u.s)
    assert u.isclose(az, 360 * u.deg)
    assert u.isclose(alt, 60.5 * u.deg)


def test_invalid_input():
    """Test invalid pointing tables raise nice errors"""
    from ctapipe.io.pointing import PointingInterpolator

    wrong_time = Table(
        {
            "time": [1, 2, 3] * u.s,
            "azimuth": [1, 2, 3] * u.deg,
            "altitude": [1, 2, 3] * u.deg,
        }
    )

    interpolator = PointingInterpolator()
    with pytest.raises(TypeError, match="astropy.time.Time"):
        interpolator.add_table(1, wrong_time)

    wrong_unit = Table(
        {
            "time": Time(1.7e9 + np.arange(3), format="unix"),
            "azimuth": [1, 2, 3] * u.m,
            "altitude": [1, 2, 3] * u.deg,
        }
    )
    with pytest.raises(ValueError, match="compatible with 'rad'"):
        interpolator.add_table(1, wrong_unit)

    wrong_unit = Table(
        {
            "time": Time(1.7e9 + np.arange(3), format="unix"),
            "azimuth": [1, 2, 3] * u.deg,
            "altitude": [1, 2, 3],
        }
    )
    with pytest.raises(ValueError, match="compatible with 'rad'"):
        interpolator.add_table(1, wrong_unit)


def test_hdf5(tmp_path):
    from ctapipe.io import write_table
    from ctapipe.io.pointing import PointingInterpolator

    t0 = Time("2022-01-01T00:00:00")

    table = Table(
        {
            "time": t0 + np.arange(0.0, 10.1, 2.0) * u.s,
            "azimuth": np.linspace(0.0, 10.0, 6) * u.deg,
            "altitude": np.linspace(70.0, 60.0, 6) * u.deg,
        },
    )

    path = tmp_path / "pointing.h5"
    write_table(table, path, "/dl0/monitoring/telescope/pointing/tel_001")
    with tables.open_file(path) as h5file:
        interpolator = PointingInterpolator(h5file)
        alt, az = interpolator(tel_id=1, time=t0 + 1 * u.s)
        assert u.isclose(alt, 69 * u.deg)
        assert u.isclose(az, 1 * u.deg)


def test_bounds():
    """Test invalid pointing tables raise nice errors"""
    from ctapipe.io.pointing import PointingInterpolator

    t0 = Time("2022-01-01T00:00:00")

    table = Table(
        {
            "time": t0 + np.arange(0.0, 10.1, 2.0) * u.s,
            "azimuth": np.linspace(0.0, 10.0, 6) * u.deg,
            "altitude": np.linspace(70.0, 60.0, 6) * u.deg,
        },
    )

    interpolator = PointingInterpolator()
    interpolator.add_table(1, table)

    with pytest.raises(ValueError, match="below the interpolation range"):
        interpolator(tel_id=1, time=t0 - 0.1 * u.s)

    with pytest.raises(ValueError, match="above the interpolation range"):
        interpolator(tel_id=1, time=t0 + 10.2 * u.s)

    interpolator = PointingInterpolator(bounds_error=False)
    interpolator.add_table(1, table)
    for dt in (-0.1, 10.1) * u.s:
        alt, az = interpolator(tel_id=1, time=t0 + dt)
        assert np.isnan(alt.value)
        assert np.isnan(az.value)

    interpolator = PointingInterpolator(bounds_error=False, extrapolate=True)
    interpolator.add_table(1, table)
    alt, az = interpolator(tel_id=1, time=t0 - 1 * u.s)
    assert u.isclose(alt, 71 * u.deg)
    assert u.isclose(az, -1 * u.deg)

    alt, az = interpolator(tel_id=1, time=t0 + 11 * u.s)
    assert u.isclose(alt, 59 * u.deg)
    assert u.isclose(az, 11 * u.deg)
