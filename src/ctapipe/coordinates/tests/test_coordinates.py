import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from pytest import approx, raises

from ctapipe.coordinates import altaz_to_righthanded_cartesian

location = EarthLocation.of_site("Roque de los Muchachos")


def test_altaz_to_righthanded_cartesian():
    """
    check the handedness of the transform
    """

    vec = altaz_to_righthanded_cartesian(alt=0 * u.deg, az=90 * u.deg)
    assert np.allclose(vec, [0, -1, 0])


def test_cam_to_nominal():
    from ctapipe.coordinates import CameraFrame, NominalFrame

    telescope_pointing = SkyCoord(alt=70 * u.deg, az=0 * u.deg, frame=AltAz())
    array_pointing = SkyCoord(alt=72 * u.deg, az=0 * u.deg, frame=AltAz())

    cam_frame = CameraFrame(
        focal_length=28 * u.m, telescope_pointing=telescope_pointing
    )
    cam = SkyCoord(x=0.5 * u.m, y=0.1 * u.m, frame=cam_frame)

    nom_frame = NominalFrame(origin=array_pointing)
    cam.transform_to(nom_frame)


def test_icrs_to_camera():
    from ctapipe.coordinates import CameraFrame

    obstime = Time("2013-11-01T03:00")
    location = EarthLocation.of_site("Roque de los Muchachos")
    horizon_frame = AltAz(location=location, obstime=obstime)

    # simulate crab "on" observations
    crab = SkyCoord(ra="05h34m31.94s", dec="22d00m52.2s")
    telescope_pointing = crab.transform_to(horizon_frame)

    camera_frame = CameraFrame(
        focal_length=28 * u.m,
        telescope_pointing=telescope_pointing,
        location=location,
        obstime=obstime,
    )

    ceta_tauri = SkyCoord(ra="5h37m38.6854231s", dec="21d08m33.158804s")
    ceta_tauri_camera = ceta_tauri.transform_to(camera_frame)

    camera_center = SkyCoord(0 * u.m, 0 * u.m, frame=camera_frame)
    crab_camera = crab.transform_to(camera_frame)

    assert crab_camera.x.to_value(u.m) == approx(0.0, abs=1e-10)
    assert crab_camera.y.to_value(u.m) == approx(0.0, abs=1e-10)

    # assert ceta tauri is in FoV
    assert camera_center.separation_3d(ceta_tauri_camera) < u.Quantity(0.6, u.m)


def test_telescope_separation():
    from ctapipe.coordinates import TelescopeFrame

    telescope_pointing = SkyCoord(alt=70 * u.deg, az=0 * u.deg, frame=AltAz())

    telescope_frame = TelescopeFrame(telescope_pointing=telescope_pointing)
    tel1 = SkyCoord(fov_lon=0 * u.deg, fov_lat=0 * u.deg, frame=telescope_frame)
    tel2 = SkyCoord(fov_lon=0 * u.deg, fov_lat=1 * u.deg, frame=telescope_frame)

    assert u.isclose(tel1.separation(tel2), 1 * u.deg)


def test_separation_is_the_same():
    from ctapipe.coordinates import TelescopeFrame

    obstime = Time("2013-11-01T03:00")
    location = EarthLocation.of_site("Roque de los Muchachos")
    horizon_frame = AltAz(location=location, obstime=obstime)

    crab = SkyCoord(ra="05h34m31.94s", dec="22d00m52.2s")
    ceta_tauri = SkyCoord(ra="5h37m38.6854231s", dec="21d08m33.158804s")

    # simulate crab "on" observations
    telescope_pointing = crab.transform_to(horizon_frame)

    telescope_frame = TelescopeFrame(
        telescope_pointing=telescope_pointing, location=location, obstime=obstime
    )

    ceta_tauri_telescope = ceta_tauri.transform_to(telescope_frame)
    crab_telescope = crab.transform_to(telescope_frame)

    sep = ceta_tauri_telescope.separation(crab_telescope).to_value(u.deg)
    assert ceta_tauri.separation(crab).to_value(u.deg) == approx(sep, rel=1e-4)


def test_cam_to_tel():
    from ctapipe.coordinates import CameraFrame, TelescopeFrame

    # Coordinates in any frame can be given as a numpy array of the xyz positions
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
    assert telescope_coord.fov_lon[0] == (1 / 15) * u.rad

    # check rotation
    camera_coord = SkyCoord(pix_x, pix_y, frame=camera_frame)
    telescope_coord_rot = camera_coord.transform_to(TelescopeFrame())
    assert telescope_coord_rot.fov_lat[0] - (1 / 15) * u.rad < 1e-6 * u.rad

    # The Transform back
    camera_coord2 = telescope_coord.transform_to(camera_frame)

    # Check separation
    assert camera_coord.separation_3d(camera_coord2)[0] == 0 * u.m


def test_cam_to_hor():
    from ctapipe.coordinates import CameraFrame

    # Coordinates in any frame can be given as a numpy array of the xyz positions
    # e.g. in this case the position on pixels in the camera
    pix_x = [1] * u.m
    pix_y = [1] * u.m

    focal_length = 15000 * u.mm

    # first define the camera frame
    pointing = SkyCoord(alt=70 * u.deg, az=0 * u.deg, frame=AltAz())
    camera_frame = CameraFrame(focal_length=focal_length, telescope_pointing=pointing)

    # transform
    camera_coord = SkyCoord(pix_x, pix_y, frame=camera_frame)
    altaz_coord = camera_coord.transform_to(AltAz())

    # transform back
    altaz_coord2 = SkyCoord(az=altaz_coord.az, alt=altaz_coord.alt, frame=AltAz())
    camera_coord2 = altaz_coord2.transform_to(camera_frame)

    # check transform
    assert np.isclose(camera_coord.x.to_value(u.m), camera_coord2.y.to_value(u.m))


def test_ground_to_tilt():
    from ctapipe.coordinates import GroundFrame, TiltedGroundFrame

    # define ground coordinate
    grd_coord = GroundFrame(x=1 * u.m, y=2 * u.m, z=0 * u.m)
    pointing_direction = SkyCoord(alt=90 * u.deg, az=0 * u.deg, frame=AltAz())

    # Convert to tilted frame at zenith (should be the same)
    tilt_coord = grd_coord.transform_to(
        TiltedGroundFrame(pointing_direction=pointing_direction)
    )
    assert tilt_coord.separation_3d(grd_coord) == 0 * u.m

    # Check 180 degree rotation reverses y coordinate
    pointing_direction = SkyCoord(alt=90 * u.deg, az=180 * u.deg, frame=AltAz())
    tilt_coord = grd_coord.transform_to(
        TiltedGroundFrame(pointing_direction=pointing_direction)
    )
    assert np.abs(tilt_coord.y + 2.0 * u.m) < 1e-5 * u.m

    # Check that if we look at horizon the x coordinate is 0
    pointing_direction = SkyCoord(alt=0 * u.deg, az=0 * u.deg, frame=AltAz())
    tilt_coord = grd_coord.transform_to(
        TiltedGroundFrame(pointing_direction=pointing_direction)
    )
    assert np.abs(tilt_coord.x) < 1e-5 * u.m


def test_ground_to_tilt_one_to_one():
    from ctapipe.coordinates import GroundFrame, TiltedGroundFrame

    # define ground coordinate
    grd_coord = GroundFrame(x=[1, 1] * u.m, y=[2, 2] * u.m, z=[0, 0] * u.m)
    pointing_direction = SkyCoord(alt=[90, 90], az=[0, 0], frame=AltAz(), unit=u.deg)

    # Convert to tilted frame at zenith (should be the same)
    tilt_coord = grd_coord.transform_to(
        TiltedGroundFrame(pointing_direction=pointing_direction)
    )

    # We do a one-to-one conversion
    assert len(tilt_coord.data) == 2


def test_ground_to_tilt_one_to_many():
    from ctapipe.coordinates import GroundFrame, TiltedGroundFrame

    # define ground coordinate
    grd_coord = GroundFrame(x=[1] * u.m, y=[2] * u.m, z=[0] * u.m)
    pointing_direction = SkyCoord(alt=[90, 90], az=[0, 0], frame=AltAz(), unit=u.deg)

    # Convert to tilted frame at zenith (should be the same)
    tilt_coord = grd_coord.transform_to(
        TiltedGroundFrame(pointing_direction=pointing_direction)
    )

    # We do a one-to-one conversion
    assert len(tilt_coord.data) == 2


def test_ground_to_tilt_many_to_one():
    from ctapipe.coordinates import GroundFrame, TiltedGroundFrame

    # define ground coordinate
    grd_coord = GroundFrame(x=[1, 1] * u.m, y=[2, 2] * u.m, z=[0, 0] * u.m)
    pointing_direction = SkyCoord(alt=90, az=0, frame=AltAz(), unit=u.deg)

    # Convert to tilted frame at zenith (should be the same)
    tilt_coord = grd_coord.transform_to(
        TiltedGroundFrame(pointing_direction=pointing_direction)
    )

    # We do a one-to-one conversion
    assert len(tilt_coord.data) == 2


def test_ground_to_tilt_many_to_many():
    from ctapipe.coordinates import GroundFrame, TiltedGroundFrame

    ground = GroundFrame(x=[1, 2] * u.m, y=[2, 1] * u.m, z=[3, 3] * u.m)
    pointing_direction = SkyCoord(
        alt=[90, 90, 90],
        az=[0, 90, 180],
        frame=AltAz(),
        unit=u.deg,
    )

    tilted = ground[:, np.newaxis].transform_to(
        TiltedGroundFrame(pointing_direction=pointing_direction)
    )

    assert tilted.shape == (2, 3)


def test_camera_missing_focal_length():
    from ctapipe.coordinates import CameraFrame, TelescopeFrame

    camera_frame = CameraFrame()
    coord = SkyCoord(x=0 * u.m, y=2 * u.m, frame=camera_frame)

    with raises(ValueError):
        coord.transform_to(TelescopeFrame())


def test_camera_focal_length_array():
    from ctapipe.coordinates import CameraFrame, TelescopeFrame

    tel_coord = SkyCoord([1, 2] * u.deg, [0, 1] * u.deg, frame=TelescopeFrame())
    cam_coord = tel_coord.transform_to(CameraFrame(focal_length=[28, 17] * u.m))
    assert not np.isnan(cam_coord.x).any()
    assert not np.isnan(cam_coord.y).any()


def test_ground_frame_roundtrip():
    """test transform from sky to ground roundtrip"""
    from ctapipe.coordinates import GroundFrame, TiltedGroundFrame

    normal = SkyCoord(alt=70 * u.deg, az=0 * u.deg, frame=AltAz())
    coord = SkyCoord(x=0, y=10, z=5, unit=u.m, frame=GroundFrame())
    tilted = coord.transform_to(TiltedGroundFrame(pointing_direction=normal))

    back = tilted.transform_to(GroundFrame())

    assert u.isclose(coord.x, back.x, atol=1e-12 * u.m)
    assert u.isclose(coord.y, back.y, atol=1e-12 * u.m)
    assert u.isclose(coord.z, back.z, atol=1e-12 * u.m)


def test_ground_to_tilt_many_to_many_roundtrip():
    from ctapipe.coordinates import GroundFrame, TiltedGroundFrame

    ground = GroundFrame(x=[1, 2] * u.m, y=[2, 1] * u.m, z=[3, 3] * u.m)
    pointing_direction = SkyCoord(
        alt=[90, 90, 90],
        az=[0, 0, 180],
        frame=AltAz(),
        unit=u.deg,
    )

    tilted = ground[:, np.newaxis].transform_to(
        TiltedGroundFrame(pointing_direction=pointing_direction)
    )
    back = tilted[:, 0].transform_to(GroundFrame())

    assert u.isclose(ground.x, back.x, atol=1e-12 * u.m).all()
    assert u.isclose(ground.y, back.y, atol=1e-12 * u.m).all()
    assert u.isclose(ground.z, back.z, atol=1e-12 * u.m).all()


def test_ground_to_eastnorth_roundtrip():
    """Check Ground to EastingNorthing and the round-trip"""
    from ctapipe.coordinates import EastingNorthingFrame, GroundFrame

    ground = SkyCoord(
        x=[1, 2, 3] * u.m, y=[-2, 5, 2] * u.m, z=[1, -1, 2] * u.m, frame=GroundFrame()
    )
    eastnorth = ground.transform_to(EastingNorthingFrame())
    ground2 = eastnorth.transform_to(GroundFrame())

    assert u.isclose(eastnorth.easting, [2, -5, -2] * u.m).all()
    assert u.isclose(eastnorth.northing, [1, 2, 3] * u.m).all()
    assert u.isclose(eastnorth.height, [1, -1, 2] * u.m).all()

    assert u.isclose(ground.x, ground2.x).all()
    assert u.isclose(ground.y, ground2.y).all()
    assert u.isclose(ground.z, ground2.z).all()
