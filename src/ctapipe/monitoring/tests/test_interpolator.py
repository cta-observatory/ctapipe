import astropy.units as u
import numpy as np
import pytest
import tables
from astropy.table import Table
from astropy.time import Time

from ctapipe.io.hdf5dataformat import DL0_TEL_POINTING_GROUP
from ctapipe.monitoring.interpolation import (
    FlatfieldImageInterpolator,
    FlatfieldPeakTimeInterpolator,
    PedestalImageInterpolator,
    PointingInterpolator,
)

t0 = Time("2022-01-01T00:00:00")


def test_chunk_selection(camera_geometry):
    table_ff = Table(
        {
            "time_start": t0 + [0, 1, 2, 6] * u.s,
            "time_end": t0 + [2, 3, 4, 8] * u.s,
            "mean": [np.full((2, len(camera_geometry)), x) for x in [1, 2, 3, 4]],
            "median": [np.full((2, len(camera_geometry)), x) for x in [1, 2, 3, 4]],
            "std": [np.full((2, len(camera_geometry)), x) for x in [1, 2, 3, 4]],
        },
    )
    interpolator_ff = FlatfieldImageInterpolator()
    interpolator_ff.add_table(1, table_ff)

    val1 = interpolator_ff(tel_id=1, time=t0 + 1.2 * u.s)
    val2 = interpolator_ff(tel_id=1, time=t0 + 1.7 * u.s)
    val3 = interpolator_ff(tel_id=1, time=t0 + 2.2 * u.s)

    for key in ["mean", "median", "std"]:
        assert np.all(np.isclose(val1[key], np.full((2, len(camera_geometry)), 2)))
        assert np.all(np.isclose(val2[key], np.full((2, len(camera_geometry)), 2)))
        assert np.all(np.isclose(val3[key], np.full((2, len(camera_geometry)), 3)))

    table_ped = Table(
        {
            "time_start": t0 + [0, 1, 2, 6] * u.s,
            "time_end": t0 + [2, 3, 4, 8] * u.s,
            "mean": [np.full((2, len(camera_geometry)), x) for x in [1, 2, 3, 4]],
            "median": [np.full((2, len(camera_geometry)), x) for x in [1, 2, 3, 4]],
            "std": [np.full((2, len(camera_geometry)), x) for x in [1, 2, 3, 4]],
        },
    )
    interpolator_ped = PedestalImageInterpolator()
    interpolator_ped.add_table(1, table_ped)

    unique_timestamps = Time([t0 + 1.2 * u.s, t0 + 1.7 * u.s, t0 + 2.2 * u.s])
    vals = interpolator_ped(tel_id=1, time=unique_timestamps)

    for key in ["mean", "median", "std"]:
        assert np.all(np.isclose(vals[key][0], np.full((2, len(camera_geometry)), 2)))
        assert np.all(np.isclose(vals[key][1], np.full((2, len(camera_geometry)), 2)))
        assert np.all(np.isclose(vals[key][2], np.full((2, len(camera_geometry)), 3)))


def test_nan_switch(camera_geometry):
    data = np.array([np.full((2, len(camera_geometry)), x) for x in [1, 2, 3, 4]])
    data[1][0, 0] = 5
    data = np.where(
        data > 4, np.nan, data
    )  # this is a workaround to introduce a nan in the data

    table_ff = Table(
        {
            "time_start": t0 + [0, 1, 2, 6] * u.s,
            "time_end": t0 + [2, 3, 4, 8] * u.s,
            "mean": data,
            "median": data,
            "std": data,
        },
    )
    interpolator_ff = FlatfieldImageInterpolator()
    interpolator_ff.add_table(1, table_ff)

    val = interpolator_ff(tel_id=1, time=t0 + 1.2 * u.s)

    res = np.full((2, len(camera_geometry)), 2)
    res[0][0] = (
        1  # where the nan was introduced before we should now have the value from the earlier chunk
    )

    for key in ["mean", "median", "std"]:
        assert np.all(np.isclose(val[key], res))

    table_ped = Table(
        {
            "time_start": t0 + [0, 1, 2, 6] * u.s,
            "time_end": t0 + [2, 3, 4, 8] * u.s,
            "mean": data,
            "median": data,
            "std": data,
        },
    )
    interpolator_ped = PedestalImageInterpolator()
    interpolator_ped.add_table(1, table_ped)

    val = interpolator_ped(tel_id=1, time=t0 + 1.2 * u.s)

    for key in ["mean", "median", "std"]:
        assert np.all(np.isclose(val[key], res))


def test_no_valid_chunk():
    table_ff = Table(
        {
            "time_start": t0 + [0, 1, 2, 6] * u.s,
            "time_end": t0 + [2, 3, 4, 8] * u.s,
            "mean": [1, 2, 3, 4],
            "median": [1, 2, 3, 4],
            "std": [1, 2, 3, 4],
        },
    )
    interpolator_ff = FlatfieldPeakTimeInterpolator()
    interpolator_ff.add_table(1, table_ff)

    val = interpolator_ff(tel_id=1, time=t0 + 5.2 * u.s)
    for key in ["mean", "median", "std"]:
        assert np.isnan(val[key])


def test_before_first_chunk():
    table_ped = Table(
        {
            "time_start": t0 + [0, 1, 2, 6] * u.s,
            "time_end": t0 + [2, 3, 4, 8] * u.s,
            "mean": [1, 2, 3, 4],
            "median": [1, 2, 3, 4],
            "std": [1, 2, 3, 4],
        },
    )
    interpolator_ped = PedestalImageInterpolator()
    interpolator_ped.add_table(1, table_ped)

    values_invalid = interpolator_ped(
        tel_id=1, time=t0 - 5.2 * u.s, timestamp_tolerance=0.25 * u.s
    )
    for key in ["mean", "median", "std"]:
        assert np.isnan(values_invalid[key])

    values_valid = interpolator_ped(
        tel_id=1, time=t0 - 0.15 * u.s, timestamp_tolerance=0.25 * u.s
    )
    for key in ["mean", "median", "std"]:
        assert values_valid[key] == table_ped[key][0]

    values_invalid = interpolator_ped(
        tel_id=1, time=t0 - 0.15 * u.s, timestamp_tolerance=0.04 * u.s
    )
    for key in ["mean", "median", "std"]:
        assert np.isnan(values_invalid[key])


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
            "azimuth": [1, 2, 3] * u.deg,
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
            "azimuth": np.linspace(0.0, 10.0, 6) * u.deg,
            "altitude": np.linspace(70.0, 60.0, 6) * u.deg,
        },
    )

    path = tmp_path / "pointing.h5"
    write_table(table, path, f"{DL0_TEL_POINTING_GROUP}/tel_001")
    with tables.open_file(path) as h5file:
        interpolator = PointingInterpolator(h5file)
        alt, az = interpolator(tel_id=1, time=t0 + 1 * u.s)
        assert u.isclose(alt, 69 * u.deg)
        assert u.isclose(az, 1 * u.deg)


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
