from astropy.coordinates import SkyCoord
from pytest import approx
import astropy.units as u


def test_roundtrip_camera_horizon():
    from ctapipe.coordinates import CameraFrame, TelescopeFrame, HorizonFrame

    telescope_pointing = SkyCoord(alt=70 * u.deg, az=0 * u.deg, frame=HorizonFrame())
    camera_frame = CameraFrame(
        focal_length=28 * u.m,
        telescope_pointing=telescope_pointing
    )

    cam_coord = SkyCoord(x=0.5 * u.m, y=0.1 * u.m, frame=camera_frame)
    telescope_coord = cam_coord.transform_to(TelescopeFrame())
    horizon_coord = telescope_coord.transform_to(HorizonFrame())

    back_telescope_coord = horizon_coord.transform_to(TelescopeFrame())
    back_cam_coord = back_telescope_coord.transform_to(camera_frame)

    delta_az = back_telescope_coord.delta_az.to_value(u.deg)
    delta_alt = back_telescope_coord.delta_alt.to_value(u.deg)
    assert delta_az == approx(telescope_coord.delta_az.to_value(u.deg))
    assert delta_alt == approx(telescope_coord.delta_alt.to_value(u.deg))

    assert back_cam_coord.x.to_value(u.m) == approx(cam_coord.x.to_value(u.m))
    assert back_cam_coord.y.to_value(u.m) == approx(cam_coord.y.to_value(u.m))
