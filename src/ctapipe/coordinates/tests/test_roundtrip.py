import astropy.units as u
from astropy.coordinates import AltAz, SkyCoord
from pytest import approx


def test_roundtrip_camera_horizon():
    from ctapipe.coordinates import CameraFrame, TelescopeFrame

    telescope_pointing = SkyCoord(alt=70 * u.deg, az=0 * u.deg, frame=AltAz())
    camera_frame = CameraFrame(
        focal_length=28 * u.m, telescope_pointing=telescope_pointing
    )

    cam_coord = SkyCoord(x=0.5 * u.m, y=0.1 * u.m, frame=camera_frame)
    telescope_coord = cam_coord.transform_to(TelescopeFrame())
    horizon_coord = telescope_coord.transform_to(AltAz())

    back_telescope_coord = horizon_coord.transform_to(TelescopeFrame())
    back_cam_coord = back_telescope_coord.transform_to(camera_frame)

    fov_lon = back_telescope_coord.fov_lon.to_value(u.deg)
    fov_lat = back_telescope_coord.fov_lat.to_value(u.deg)
    assert fov_lon == approx(telescope_coord.fov_lon.to_value(u.deg))
    assert fov_lat == approx(telescope_coord.fov_lat.to_value(u.deg))

    assert back_cam_coord.x.to_value(u.m) == approx(cam_coord.x.to_value(u.m))
    assert back_cam_coord.y.to_value(u.m) == approx(cam_coord.y.to_value(u.m))
