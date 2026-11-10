import warnings

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import AltAz


def test_point_on_shower_axis_far(subarray_prod5_paranal):
    """Test for get_point_on_shower_axis"""
    from ctapipe.coordinates import get_point_on_shower_axis

    core_x = 50 * u.m
    core_y = 100 * u.m
    alt = 68 * u.deg
    az = 30 * u.deg
    # for a very large distance, should be identical to the shower direction
    distance = 1000 * u.km

    point = get_point_on_shower_axis(
        core_x=core_x,
        core_y=core_y,
        alt=alt,
        az=az,
        telescope_position=subarray_prod5_paranal.tel_coords,
        distance=distance,
    )

    np.testing.assert_allclose(point.alt, alt, rtol=1e-3)
    np.testing.assert_allclose(point.az, az, rtol=1e-2)


def test_single_telescope(subarray_prod5_paranal):
    from ctapipe.coordinates import (
        MissingFrameAttributeWarning,
        get_point_on_shower_axis,
    )

    core_x = 50 * u.m
    core_y = 100 * u.m
    alt = 68 * u.deg
    az = 30 * u.deg
    distance = 10 * u.km

    point = get_point_on_shower_axis(
        core_x=core_x,
        core_y=core_y,
        alt=alt,
        az=az,
        telescope_position=subarray_prod5_paranal.tel_coords[0],
        distance=distance,
    )

    source = AltAz(alt=alt, az=az)
    # 10 km is around the shower maximum, should be around 1 degree from the source
    with pytest.warns(MissingFrameAttributeWarning):
        assert u.isclose(source.separation(point), 1.0 * u.deg, atol=0.1 * u.deg)


def test_altaz_to_nominal():
    from ctapipe.coordinates import NominalFrame, altaz_to_nominal

    coord = altaz_to_nominal(
        az=[220.0, 220.2] * u.deg,
        alt=[80.0, 79.2] * u.deg,
        pointing_az=[220.0, 220.0] * u.deg,
        pointing_alt=[80.0, 80.0] * u.deg,
    )

    assert isinstance(coord.frame, NominalFrame)
    assert u.allclose(coord.fov_lon, [0, 0.03747984] * u.deg, atol=1e-8 * u.deg)
    assert u.allclose(coord.fov_lat, [0, -0.79993558] * u.deg, atol=1e-8 * u.deg)


@pytest.fixture
def altaz_to_icrs_inputs():
    """Three events at different times, seen from the CTAO southern site."""
    from astropy.coordinates import EarthLocation
    from astropy.time import Time

    return dict(
        az=[271.0, 268.5, 270.5] * u.deg,
        alt=[70.1, 69.5, 70.8] * u.deg,
        obstime=Time("2020-06-20T00:00:00") + [0, 10, 20] * u.s,
        location=EarthLocation(lon=-70.3 * u.deg, lat=-24.6 * u.deg, height=2150 * u.m),
    )


def test_altaz_to_icrs(altaz_to_icrs_inputs):
    """Check the result against an exact astropy transformation."""
    from astropy.coordinates import ICRS, SkyCoord

    from ctapipe.coordinates import altaz_to_icrs

    expected = SkyCoord(frame="altaz", **altaz_to_icrs_inputs).icrs

    result = altaz_to_icrs(**altaz_to_icrs_inputs)

    assert isinstance(result.frame, ICRS)
    assert result.shape == (3,)
    assert u.allclose(result.ra, expected.ra, atol=1 * u.arcsec)
    assert u.allclose(result.dec, expected.dec, atol=1 * u.arcsec)

    # extra frame attributes are kept, so the result can be transformed back
    assert np.all(result.obstime == altaz_to_icrs_inputs["obstime"])
    assert result.location == altaz_to_icrs_inputs["location"]
    altaz = result.transform_to("altaz")
    assert u.allclose(altaz.az, altaz_to_icrs_inputs["az"], atol=1 * u.arcsec)
    assert u.allclose(altaz.alt, altaz_to_icrs_inputs["alt"], atol=1 * u.arcsec)


def test_altaz_to_icrs_non_finite(altaz_to_icrs_inputs):
    """Check non-finite events give NaN, without affecting or warning about others."""
    from ctapipe.coordinates import altaz_to_icrs

    expected = altaz_to_icrs(**altaz_to_icrs_inputs)

    az = altaz_to_icrs_inputs["az"].copy()
    alt = altaz_to_icrs_inputs["alt"].copy()
    az[1] = np.nan
    alt[2] = np.inf
    inputs = altaz_to_icrs_inputs | dict(az=az, alt=alt)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = altaz_to_icrs(**inputs)

    assert u.isclose(result[0].ra, expected[0].ra)
    assert u.isclose(result[0].dec, expected[0].dec)
    assert np.all(np.isnan(result[1:].ra))
    assert np.all(np.isnan(result[1:].dec))

    # nothing valid
    inputs = altaz_to_icrs_inputs | dict(az=np.full(3, np.nan) * u.deg)
    result = altaz_to_icrs(**inputs)
    assert result.shape == (3,)
    assert np.all(np.isnan(result.ra))
    assert np.all(np.isnan(result.dec))


def test_altaz_to_icrs_scalars(altaz_to_icrs_inputs):
    """Check scalar inputs are broadcast and give the same result as arrays."""
    from ctapipe.coordinates import altaz_to_icrs

    inputs = altaz_to_icrs_inputs
    az, alt, obstime = inputs["az"], inputs["alt"], inputs["obstime"]
    location = inputs["location"]

    def assert_same_coord(result, expected):
        assert result.shape == expected.shape
        assert u.allclose(result.ra, expected.ra)
        assert u.allclose(result.dec, expected.dec)

    # all scalars give a scalar coordinate, including a non-finite event
    expected = altaz_to_icrs(az, alt, obstime, location)
    result = altaz_to_icrs(az[0], alt[0], obstime[0], location)
    assert result.isscalar
    assert_same_coord(result, expected[0])

    result = altaz_to_icrs(np.nan * u.deg, alt[0], obstime[0], location)
    assert result.isscalar
    assert np.isnan(result.ra) and np.isnan(result.dec)

    # a scalar time with array coordinates
    expected = altaz_to_icrs(az, alt, obstime[[0, 0, 0]], location)
    assert_same_coord(altaz_to_icrs(az, alt, obstime[0], location), expected)

    # scalar coordinates with an array of times
    expected = altaz_to_icrs(az[[0, 0, 0]], alt[[0, 0, 0]], obstime, location)
    assert_same_coord(altaz_to_icrs(az[0], alt[0], obstime, location), expected)
