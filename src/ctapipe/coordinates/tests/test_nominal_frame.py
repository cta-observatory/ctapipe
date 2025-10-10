import astropy.units as u
from astropy.coordinates import AltAz, SkyCoord


def test_nominal_to_horizontal_alt0_az0():
    from ctapipe.coordinates.nominal_frame import NominalFrame

    horizon_frame = AltAz()
    pointing = SkyCoord(az=0 * u.deg, alt=0 * u.deg, frame=horizon_frame)

    nominal_frame = NominalFrame(origin=pointing)

    nominal_coord = SkyCoord(fov_lon=1 * u.deg, fov_lat=0 * u.deg, frame=nominal_frame)
    horizon_coord = nominal_coord.transform_to(horizon_frame)
    assert u.isclose(horizon_coord.az, 1.0 * u.deg)
    assert u.isclose(horizon_coord.alt, 0.0 * u.deg)

    nominal_coord = SkyCoord(fov_lon=-1 * u.deg, fov_lat=0 * u.deg, frame=nominal_frame)
    horizon_coord = nominal_coord.transform_to(horizon_frame)
    assert u.isclose(horizon_coord.az.wrap_at("180d"), -1.0 * u.deg)
    assert u.isclose(horizon_coord.alt, 0.0 * u.deg)

    nominal_coord = SkyCoord(fov_lon=0 * u.deg, fov_lat=1 * u.deg, frame=nominal_frame)
    horizon_coord = nominal_coord.transform_to(horizon_frame)
    assert u.isclose(horizon_coord.az, 0.0 * u.deg)
    assert u.isclose(horizon_coord.alt, 1.0 * u.deg)

    nominal_coord = SkyCoord(fov_lon=0 * u.deg, fov_lat=-1 * u.deg, frame=nominal_frame)
    horizon_coord = nominal_coord.transform_to(horizon_frame)
    assert u.isclose(horizon_coord.az, 0.0 * u.deg)
    assert u.isclose(horizon_coord.alt, -1.0 * u.deg)


def test_nominal_to_horizontal_alt0_az180():
    from ctapipe.coordinates.nominal_frame import NominalFrame

    horizon_frame = AltAz()
    pointing = SkyCoord(az=180 * u.deg, alt=0 * u.deg, frame=horizon_frame)

    nominal_frame = NominalFrame(origin=pointing)

    nominal_coord = SkyCoord(fov_lon=1 * u.deg, fov_lat=0 * u.deg, frame=nominal_frame)
    horizon_coord = nominal_coord.transform_to(horizon_frame)
    assert u.isclose(horizon_coord.az, 181.0 * u.deg)
    assert u.isclose(horizon_coord.alt, 0.0 * u.deg)

    nominal_coord = SkyCoord(fov_lon=-1 * u.deg, fov_lat=0 * u.deg, frame=nominal_frame)
    horizon_coord = nominal_coord.transform_to(horizon_frame)
    assert u.isclose(horizon_coord.az, 179.0 * u.deg)
    assert u.isclose(horizon_coord.alt, 0.0 * u.deg)

    nominal_coord = SkyCoord(fov_lon=0 * u.deg, fov_lat=1 * u.deg, frame=nominal_frame)
    horizon_coord = nominal_coord.transform_to(horizon_frame)
    assert u.isclose(horizon_coord.az, 180.0 * u.deg)
    assert u.isclose(horizon_coord.alt, 1.0 * u.deg)

    nominal_coord = SkyCoord(fov_lon=0 * u.deg, fov_lat=-1 * u.deg, frame=nominal_frame)
    horizon_coord = nominal_coord.transform_to(horizon_frame)
    assert u.isclose(horizon_coord.az, 180.0 * u.deg)
    assert u.isclose(horizon_coord.alt, -1.0 * u.deg)
