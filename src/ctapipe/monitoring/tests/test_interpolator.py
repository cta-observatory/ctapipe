import astropy.units as u
import numpy as np
import pytest
import tables
from astropy.table import Table
from astropy.time import Time

from ctapipe.monitoring.interpolation import ChunkInterpolator, PointingInterpolator

t0 = Time("2022-01-01T00:00:00")


def test_chunk_selection():
    table = Table(
        {
            "start_time": t0 + [0, 1, 2, 6] * u.s,
            "end_time": t0 + [2, 3, 4, 8] * u.s,
            "values": [1, 2, 3, 4],
        },
    )
    interpolator = ChunkInterpolator()
    interpolator.add_table(1, table, ["values"])

    val1 = interpolator(tel_id=1, time=t0 + 1.2 * u.s, columns="values")
    val2 = interpolator(tel_id=1, time=t0 + 1.7 * u.s, columns="values")
    val3 = interpolator(tel_id=1, time=t0 + 2.2 * u.s, columns="values")

    assert np.isclose(val1, 2)
    assert np.isclose(val2, 2)
    assert np.isclose(val3, 3)


def test_chunk_selection_multiple_columns():
    table = Table(
        {
            "start_time": t0 + [0, 1, 2, 6] * u.s,
            "end_time": t0 + [2, 3, 4, 8] * u.s,
            "values1": [1, 2, 3, 4],
            "values2": [10, 20, 30, 40],
        },
    )
    interpolator = ChunkInterpolator()
    interpolator.add_table(1, table, ["values1", "values2"])

    result1 = interpolator(
        tel_id=1, time=t0 + 1.2 * u.s, columns=["values1", "values2"]
    )
    result2 = interpolator(
        tel_id=1, time=t0 + 1.7 * u.s, columns=["values1", "values2"]
    )
    result3 = interpolator(
        tel_id=1, time=t0 + 2.2 * u.s, columns=["values1", "values2"]
    )

    assert np.isclose(result1["values1"], 2)
    assert np.isclose(result1["values2"], 20)
    assert np.isclose(result2["values1"], 2)
    assert np.isclose(result2["values2"], 20)
    assert np.isclose(result3["values1"], 3)
    assert np.isclose(result3["values2"], 30)


def test_nan_switch():
    table = Table(
        {
            "start_time": t0 + [0, 1, 2, 6] * u.s,
            "end_time": t0 + [2, 3, 4, 8] * u.s,
            "values": [1, np.nan, 3, 4],
        },
    )
    interpolator = ChunkInterpolator()
    interpolator.add_table(1, table, ["values"])

    val = interpolator(tel_id=1, time=t0 + 1.2 * u.s, columns="values")

    assert np.isclose(val, 1)


def test_nan_switch_multiple_columns():
    table = Table(
        {
            "start_time": t0 + [0, 1, 2, 6] * u.s,
            "end_time": t0 + [2, 3, 4, 8] * u.s,
            "values1": [1, np.nan, 3, 4],
            "values2": [10, 20, np.nan, 40],
        },
    )
    interpolator = ChunkInterpolator()
    interpolator.add_table(1, table, ["values1", "values2"])

    result = interpolator(tel_id=1, time=t0 + 1.2 * u.s, columns=["values1", "values2"])

    assert np.isclose(result["values1"], 1)
    assert np.isclose(result["values2"], 20)


def test_no_valid_chunk():
    table = Table(
        {
            "start_time": t0 + [0, 1, 2, 6] * u.s,
            "end_time": t0 + [2, 3, 4, 8] * u.s,
            "values": [1, 2, 3, 4],
        },
    )
    interpolator = ChunkInterpolator()
    interpolator.add_table(1, table, ["values"])

    val = interpolator(tel_id=1, time=t0 + 5.2 * u.s, columns="values")
    assert np.isnan(val)


def test_no_valid_chunk_multiple_columns():
    table = Table(
        {
            "start_time": t0 + [0, 1, 2, 6] * u.s,
            "end_time": t0 + [2, 3, 4, 8] * u.s,
            "values1": [1, 2, 3, 4],
            "values2": [10, 20, 30, 40],
        },
    )
    interpolator = ChunkInterpolator()
    interpolator.add_table(1, table, ["values1", "values2"])

    result = interpolator(tel_id=1, time=t0 + 5.2 * u.s, columns=["values1", "values2"])
    assert np.isnan(result["values1"])
    assert np.isnan(result["values2"])


def test_azimuth_switchover():
    """Test pointing interpolation"""

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
            "azimuth": np.radians([1, 2, 3]) * u.rad,
            "altitude": [1, 2, 3],
        }
    )
    with pytest.raises(ValueError, match="compatible with 'rad'"):
        interpolator.add_table(1, wrong_unit)


def test_hdf5(tmp_path):
    """Test writing interpolated data to file"""
    from ctapipe.io import write_table

    table = Table(
        {
            "time": t0 + np.arange(0.0, 10.1, 2.0) * u.s,
            "azimuth": np.radians(np.linspace(0.0, 10.0, 6)) * u.rad,
            "altitude": np.radians(np.linspace(70.0, 60.0, 6)) * u.rad,
        },
    )

    path = tmp_path / "pointing.h5"
    write_table(table, path, "/dl0/monitoring/telescope/pointing/tel_001")
    with tables.open_file(path) as h5file:
        interpolator = PointingInterpolator(h5file)
        alt, az = interpolator(tel_id=1, time=t0 + 1 * u.s)
        assert u.isclose(alt, np.radians(69) * u.rad)
        assert u.isclose(az, np.radians(1) * u.rad)


def test_bounds():
    """Test invalid pointing tables raise nice errors"""

    table_pointing = Table(
        {
            "time": t0 + np.arange(0.0, 10.1, 2.0) * u.s,
            "azimuth": np.linspace(0.0, 10.0, 6) * u.deg,
            "altitude": np.linspace(70.0, 60.0, 6) * u.deg,
        },
    )

    interpolator_pointing = PointingInterpolator()
    interpolator_pointing.add_table(1, table_pointing)
    error_message = "below the interpolation range"

    with pytest.raises(ValueError, match=error_message):
        interpolator_pointing(tel_id=1, time=t0 - 0.1 * u.s)

    with pytest.raises(ValueError, match="above the interpolation range"):
        interpolator_pointing(tel_id=1, time=t0 + 10.2 * u.s)

    alt, az = interpolator_pointing(tel_id=1, time=t0 + 1 * u.s)
    assert u.isclose(alt, 69 * u.deg)
    assert u.isclose(az, 1 * u.deg)

    with pytest.raises(KeyError):
        interpolator_pointing(tel_id=2, time=t0 + 1 * u.s)

    interpolator_pointing = PointingInterpolator(bounds_error=False)
    interpolator_pointing.add_table(1, table_pointing)
    for dt in (-0.1, 10.1) * u.s:
        alt, az = interpolator_pointing(tel_id=1, time=t0 + dt)
        assert np.isnan(alt.value)
        assert np.isnan(az.value)

    interpolator_pointing = PointingInterpolator(bounds_error=False, extrapolate=True)
    interpolator_pointing.add_table(1, table_pointing)
    alt, az = interpolator_pointing(tel_id=1, time=t0 - 1 * u.s)
    assert u.isclose(alt, 71 * u.deg)
    assert u.isclose(az, -1 * u.deg)

    alt, az = interpolator_pointing(tel_id=1, time=t0 + 11 * u.s)
    assert u.isclose(alt, 59 * u.deg)
    assert u.isclose(az, 11 * u.deg)
