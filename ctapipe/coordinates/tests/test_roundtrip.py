from astropy.coordinates import SkyCoord
import astropy.units as u


def test_roundtrip_camera_horizon():
    from ctapipe.coordinates import CameraFrame, TelescopeFrame, HorizonFrame

    telescope_pointing = SkyCoord(alt=70 * u.deg, az=0 * u.deg, frame=HorizonFrame())
    camera_frame = CameraFrame(focal_length=28 * u.m, pointing_direction=telescope_pointing)

    cam_coord = SkyCoord(x=0.5 * u.m, y=0.1 * u.m, frame=camera_frame)
    telescope_coord = cam_coord.transform_to(TelescopeFrame())
    horizon_coord = telescope_coord.transform_to(HorizonFrame())

    back_telescope_coord = horizon_coord.transform_to(TelescopeFrame())
    back_cam_coord = back_telescope_coord.transform_to(camera_frame)

    assert back_telescope_coord.x == telescope_coord.x
    assert back_telescope_coord.y == telescope_coord.y

    assert back_cam_coord.x == cam_coord.x
    assert back_cam_coord.y == cam_coord.y
