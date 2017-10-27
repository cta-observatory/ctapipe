import numpy as np
import astropy.units as u


def test_cam_to_tel():
    from ctapipe.coordinates import CameraFrame, TelescopeFrame

    # Coordinates in any fram can be given as a numpy array of the xyz positions
    # e.g. in this case the position on pixels in the camera
    pix_x = [1] * u.m
    pix_y = [1] * u.m

    focal_length = 15 * u.m

    # first define the camera frame
    camera_coord = CameraFrame(pix_x, pix_y, focal_length=focal_length)

    # then use transform to function to convert to a new system
    # making sure to give the required values for the conversion
    # (these are not checked yet)
    telescope_coord = camera_coord.transform_to(TelescopeFrame())
    assert telescope_coord.x[0] == (1 / 15) * u.rad

    # check rotation
    camera_coord = CameraFrame(pix_x, pix_y, focal_length=focal_length)
    telescope_coord_rot = camera_coord.transform_to(TelescopeFrame())
    assert telescope_coord_rot.y[0] - (1 / 15) * u.rad < 1e-6 * u.rad

    # The Transform back
    camera_coord2 = telescope_coord.transform_to(
        CameraFrame(focal_length=focal_length)
    )

    # Check separation
    assert camera_coord.separation_3d(camera_coord2)[0] == 0 * u.m


def test_ground_to_tilt():
    from ctapipe.coordinates import GroundFrame, TiltedGroundFrame, HorizonFrame

    # define ground coordinate
    grd_coord = GroundFrame(x=1 * u.m, y=2 * u.m, z=0 * u.m)
    pointing_direction = HorizonFrame(alt=90 * u.deg, az=0 * u.deg)

    # Convert to tilted frame at zenith (should be the same)
    tilt_coord = grd_coord.transform_to(
        TiltedGroundFrame(pointing_direction=pointing_direction)
    )
    assert tilt_coord.separation_3d(grd_coord) == 0 * u.m

    # Check 180 degree rotation reverses y coordinate
    pointing_direction = HorizonFrame(alt=90 * u.deg, az=180 * u.deg)
    tilt_coord = grd_coord.transform_to(
        TiltedGroundFrame(pointing_direction=pointing_direction)
    )
    assert np.abs(tilt_coord.y + 2. * u.m) < 1e-5 * u.m

    # Check that if we look at horizon the x coordinate is 0
    pointing_direction = HorizonFrame(alt=0 * u.deg, az=0 * u.deg)
    tilt_coord = grd_coord.transform_to(
        TiltedGroundFrame(pointing_direction=pointing_direction)
    )
    assert np.abs(tilt_coord.x) < 1e-5 * u.m
