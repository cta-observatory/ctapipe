import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
from pytest import approx

location = EarthLocation.of_site('Roque de los Muchachos')


def test_cam_to_nominal():
    from ctapipe.coordinates import CameraFrame, HorizonFrame, NominalFrame

    telescope_pointing = SkyCoord(alt=70 * u.deg, az=0 * u.deg, frame=HorizonFrame())
    array_pointing = SkyCoord(alt=72 * u.deg, az=0 * u.deg, frame=HorizonFrame())

    cam_frame = CameraFrame(focal_length=28 * u.m, telescope_pointing=telescope_pointing)
    cam = SkyCoord(x=0.5 * u.m, y=0.1 * u.m, frame=cam_frame)

    nom_frame = NominalFrame(origin=array_pointing)
    cam.transform_to(nom_frame)


def test_icrs_to_camera():
    from ctapipe.coordinates import CameraFrame, HorizonFrame

    obstime = Time('2013-11-01T03:00')
    location = EarthLocation.of_site('Roque de los Muchachos')
    horizon_frame = HorizonFrame(location=location, obstime=obstime)

    # simulate crab "on" observations
    crab = SkyCoord(ra='05h34m31.94s', dec='22d00m52.2s')
    telescope_pointing = crab.transform_to(horizon_frame)

    camera_frame = CameraFrame(
        focal_length=28 * u.m,
        telescope_pointing=telescope_pointing,
        location=location, obstime=obstime,
    )

    ceta_tauri = SkyCoord(ra='5h37m38.6854231s', dec='21d08m33.158804s')
    ceta_tauri_camera = ceta_tauri.transform_to(camera_frame)

    camera_center = SkyCoord(0 * u.m, 0 * u.m, frame=camera_frame)
    crab_camera = crab.transform_to(camera_frame)

    assert crab_camera.x.to_value(u.m) == approx(0.0, abs=1e-10)
    assert crab_camera.y.to_value(u.m) == approx(0.0, abs=1e-10)

    # assert ceta tauri is in FoV
    assert camera_center.separation_3d(ceta_tauri_camera) < u.Quantity(0.6, u.m)


def test_telescope_separation():
    from ctapipe.coordinates import TelescopeFrame, HorizonFrame

    telescope_pointing = SkyCoord(
        alt=70 * u.deg,
        az=0 * u.deg,
        frame=HorizonFrame()
    )

    telescope_frame = TelescopeFrame(telescope_pointing=telescope_pointing)
    tel1 = SkyCoord(
        delta_az=0 * u.deg, delta_alt=0 * u.deg, frame=telescope_frame
    )
    tel2 = SkyCoord(
        delta_az=0 * u.deg, delta_alt=1 * u.deg, frame=telescope_frame
    )

    assert tel1.separation(tel2) == u.Quantity(1, u.deg)


def test_separation_is_the_same():
    from ctapipe.coordinates import TelescopeFrame, HorizonFrame

    obstime = Time('2013-11-01T03:00')
    location = EarthLocation.of_site('Roque de los Muchachos')
    horizon_frame = HorizonFrame(location=location, obstime=obstime)

    crab = SkyCoord(ra='05h34m31.94s', dec='22d00m52.2s')
    ceta_tauri = SkyCoord(ra='5h37m38.6854231s', dec='21d08m33.158804s')

    # simulate crab "on" observations
    telescope_pointing = crab.transform_to(horizon_frame)

    telescope_frame = TelescopeFrame(
        telescope_pointing=telescope_pointing,
        location=location,
        obstime=obstime,
    )

    ceta_tauri_telescope = ceta_tauri.transform_to(telescope_frame)
    crab_telescope = crab.transform_to(telescope_frame)

    sep = ceta_tauri_telescope.separation(crab_telescope).to_value(u.deg)
    assert ceta_tauri.separation(crab).to_value(u.deg) == approx(sep, rel=1e-4)


def test_cam_to_tel():
    from ctapipe.coordinates import CameraFrame, TelescopeFrame

    # Coordinates in any fram can be given as a numpy array of the xyz positions
    # e.g. in this case the position on pixels in the camera
    pix_x = [1] * u.m
    pix_y = [1] * u.m

    focal_length = 15 * u.m

    camera_frame = CameraFrame(focal_length=focal_length)
    # first define the camera frame
    camera_coord = SkyCoord(pix_x, pix_y, frame=camera_frame)

    # then use transform to function to convert to a new system
    # making sure to give the required values for the conversion
    # (these are not checked yet)
    telescope_coord = camera_coord.transform_to(TelescopeFrame())
    assert telescope_coord.delta_az[0] == (1 / 15) * u.rad

    # check rotation
    camera_coord = SkyCoord(pix_x, pix_y, frame=camera_frame)
    telescope_coord_rot = camera_coord.transform_to(TelescopeFrame())
    assert telescope_coord_rot.delta_alt[0] - (1 / 15) * u.rad < 1e-6 * u.rad

    # The Transform back
    camera_coord2 = telescope_coord.transform_to(camera_frame)

    # Check separation
    assert camera_coord.separation_3d(camera_coord2)[0] == 0 * u.m


def test_ground_to_tilt():
    from ctapipe.coordinates import GroundFrame, TiltedGroundFrame, HorizonFrame

    # define ground coordinate
    grd_coord = GroundFrame(x=1 * u.m, y=2 * u.m, z=0 * u.m)
    pointing_direction = SkyCoord(alt=90 * u.deg, az=0 * u.deg, frame=HorizonFrame())

    # Convert to tilted frame at zenith (should be the same)
    tilt_coord = grd_coord.transform_to(
        TiltedGroundFrame(pointing_direction=pointing_direction)
    )
    assert tilt_coord.separation_3d(grd_coord) == 0 * u.m

    # Check 180 degree rotation reverses y coordinate
    pointing_direction = SkyCoord(alt=90 * u.deg, az=180 * u.deg, frame=HorizonFrame())
    tilt_coord = grd_coord.transform_to(
        TiltedGroundFrame(pointing_direction=pointing_direction)
    )
    assert np.abs(tilt_coord.y + 2. * u.m) < 1e-5 * u.m

    # Check that if we look at horizon the x coordinate is 0
    pointing_direction = SkyCoord(alt=0 * u.deg, az=0 * u.deg, frame=HorizonFrame())
    tilt_coord = grd_coord.transform_to(
        TiltedGroundFrame(pointing_direction=pointing_direction)
    )
    assert np.abs(tilt_coord.x) < 1e-5 * u.m
