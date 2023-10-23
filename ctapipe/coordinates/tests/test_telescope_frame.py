import astropy.units as u
from astropy.coordinates import AltAz, SkyCoord
from pytest import approx


def test_telescope_to_horizontal_alt0_az0():
    from ctapipe.coordinates.telescope_frame import TelescopeFrame

    horizon_frame = AltAz()
    pointing = SkyCoord(az=0 * u.deg, alt=0 * u.deg, frame=horizon_frame)

    telescope_frame = TelescopeFrame(telescope_pointing=pointing)

    telescope_coord = SkyCoord(
        fov_lon=1 * u.deg, fov_lat=0 * u.deg, frame=telescope_frame
    )
    horizon_coord = telescope_coord.transform_to(horizon_frame)
    assert horizon_coord.az.deg == 1.0
    assert horizon_coord.alt.deg == 0.0

    telescope_coord = SkyCoord(
        fov_lon=-1 * u.deg, fov_lat=0 * u.deg, frame=telescope_frame
    )
    horizon_coord = telescope_coord.transform_to(horizon_frame)
    assert horizon_coord.az.wrap_at("180d").deg == -1.0
    assert horizon_coord.alt.deg == 0.0

    telescope_coord = SkyCoord(
        fov_lon=0 * u.deg, fov_lat=1 * u.deg, frame=telescope_frame
    )
    horizon_coord = telescope_coord.transform_to(horizon_frame)
    assert horizon_coord.az.deg == 0.0
    assert horizon_coord.alt.deg == 1.0

    telescope_coord = SkyCoord(
        fov_lon=0 * u.deg, fov_lat=-1 * u.deg, frame=telescope_frame
    )
    horizon_coord = telescope_coord.transform_to(horizon_frame)
    assert horizon_coord.az.deg == 0.0
    assert horizon_coord.alt.deg == -1.0


def test_telescope_to_horizontal_alt0_az180():
    from ctapipe.coordinates.telescope_frame import TelescopeFrame

    horizon_frame = AltAz()
    pointing = SkyCoord(az=180 * u.deg, alt=0 * u.deg, frame=horizon_frame)

    telescope_frame = TelescopeFrame(telescope_pointing=pointing)

    telescope_coord = SkyCoord(
        fov_lon=1 * u.deg, fov_lat=0 * u.deg, frame=telescope_frame
    )
    horizon_coord = telescope_coord.transform_to(horizon_frame)
    assert horizon_coord.az.deg == approx(181.0)
    assert horizon_coord.alt.deg == 0.0

    telescope_coord = SkyCoord(
        fov_lon=-1 * u.deg, fov_lat=0 * u.deg, frame=telescope_frame
    )
    horizon_coord = telescope_coord.transform_to(horizon_frame)
    assert horizon_coord.az.deg == approx(179.0)
    assert horizon_coord.alt.deg == 0.0

    telescope_coord = SkyCoord(
        fov_lon=0 * u.deg, fov_lat=1 * u.deg, frame=telescope_frame
    )
    horizon_coord = telescope_coord.transform_to(horizon_frame)
    assert horizon_coord.az.deg == 180.0
    assert horizon_coord.alt.deg == 1.0

    telescope_coord = SkyCoord(
        fov_lon=0 * u.deg, fov_lat=-1 * u.deg, frame=telescope_frame
    )
    horizon_coord = telescope_coord.transform_to(horizon_frame)
    assert horizon_coord.az.deg == 180.0
    assert horizon_coord.alt.deg == -1.0
