"""

"""
import astropy.units as u
import pytest
from astropy.coordinates import EarthLocation

# LST-1, MAGIC-1, MAGIC-2
locations = [
    EarthLocation(lon=-17.89149701 * u.deg, lat=28.76152611 * u.deg, height=2185 * u.m),
    EarthLocation(
        lon=-17.8900665372 * u.deg, lat=28.7619439944 * u.deg, height=2184.763 * u.m
    ),
    EarthLocation(
        lon=-17.8905595556 * u.deg, lat=28.7613094808 * u.deg, height=2186.764 * u.m
    ),
]

# dish-area weighted center
reference_location = EarthLocation(
    lon=-17.890879 * u.deg,
    lat=28.761579 * u.deg,
    height=2199 * u.m,
)

# these are the current positions in the current simtel-config,
# they used a slightly different inputs, so results are not expected to be
# exactly equal
mc_positions = [
    [-6.336, 60.405, 12.5],
    [41.054, -79.275, 10.75],
    [-29.456001, -31.295, 11.92],
] * u.m


@pytest.mark.parametrize("location,expected", zip(locations, mc_positions))
def test_ground_frame_to_earth_location(location, expected):
    from ctapipe.coordinates import GroundFrame

    ground_frame = GroundFrame.from_earth_location(location, reference_location)
    x, y, z = ground_frame.cartesian.xyz
    ex, ey, _ = expected

    # the calculation for the mc position used slightly different inputs
    # so we accept a bit of a tolerance here
    assert u.isclose(x, ex, atol=0.6 * u.m)
    assert u.isclose(y, ey, atol=0.6 * u.m)

    # the mc positions are that of the center of the dish and were shifted
    # by simtel, so we compare our expectation here and not to the simtel values
    assert u.isclose(z, location.height - reference_location.height, atol=0.1 * u.m)

    # test back_trafo
    back = ground_frame.to_earth_location()
    assert u.isclose(back.lon, location.lon)
    assert u.isclose(back.lat, location.lat)
    assert u.isclose(back.height, location.height)
