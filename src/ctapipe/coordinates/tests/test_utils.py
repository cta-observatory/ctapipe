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
