from astropy.coordinates import SkyCoord, AltAz
import astropy.units as u
from pytest import approx


def test_nominal_to_horizontal_alt0_az0():
    from ctapipe.coordinates.nominal_frame import NominalFrame

    horizon_frame = AltAz()
    pointing = SkyCoord(az=0 * u.deg, alt=0 * u.deg, frame=horizon_frame)

    nominal_frame = NominalFrame(origin=pointing)

    nominal_coord = SkyCoord(fov_lon=1 * u.deg, fov_lat=0 * u.deg, frame=nominal_frame)
    horizon_coord = nominal_coord.transform_to(horizon_frame)
    assert horizon_coord.az.deg == 1.0
    assert horizon_coord.alt.deg == 0.0

    nominal_coord = SkyCoord(fov_lon=-1 * u.deg, fov_lat=0 * u.deg, frame=nominal_frame)
    horizon_coord = nominal_coord.transform_to(horizon_frame)
    assert horizon_coord.az.wrap_at("180d").deg == -1.0
    assert horizon_coord.alt.deg == 0.0

    nominal_coord = SkyCoord(fov_lon=0 * u.deg, fov_lat=1 * u.deg, frame=nominal_frame)
    horizon_coord = nominal_coord.transform_to(horizon_frame)
    assert horizon_coord.az.deg == 0.0
    assert horizon_coord.alt.deg == 1.0

    nominal_coord = SkyCoord(fov_lon=0 * u.deg, fov_lat=-1 * u.deg, frame=nominal_frame)
    horizon_coord = nominal_coord.transform_to(horizon_frame)
    assert horizon_coord.az.deg == 0.0
    assert horizon_coord.alt.deg == -1.0


def test_nominal_to_horizontal_alt0_az180():
    from ctapipe.coordinates.nominal_frame import NominalFrame

    horizon_frame = AltAz()
    pointing = SkyCoord(az=180 * u.deg, alt=0 * u.deg, frame=horizon_frame)

    nominal_frame = NominalFrame(origin=pointing)

    nominal_coord = SkyCoord(fov_lon=1 * u.deg, fov_lat=0 * u.deg, frame=nominal_frame)
    horizon_coord = nominal_coord.transform_to(horizon_frame)
    assert horizon_coord.az.deg == approx(181.0)
    assert horizon_coord.alt.deg == 0.0

    nominal_coord = SkyCoord(fov_lon=-1 * u.deg, fov_lat=0 * u.deg, frame=nominal_frame)
    horizon_coord = nominal_coord.transform_to(horizon_frame)
    assert horizon_coord.az.deg == approx(179.0)
    assert horizon_coord.alt.deg == 0.0

    nominal_coord = SkyCoord(fov_lon=0 * u.deg, fov_lat=1 * u.deg, frame=nominal_frame)
    horizon_coord = nominal_coord.transform_to(horizon_frame)
    assert horizon_coord.az.deg == 180.0
    assert horizon_coord.alt.deg == 1.0

    nominal_coord = SkyCoord(fov_lon=0 * u.deg, fov_lat=-1 * u.deg, frame=nominal_frame)
    horizon_coord = nominal_coord.transform_to(horizon_frame)
    assert horizon_coord.az.deg == 180.0
    assert horizon_coord.alt.deg == -1.0
