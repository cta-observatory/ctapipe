from astropy.coordinates import SkyCoord
import astropy.units as u


def test_conversion():
    from ctapipe.coordinates import CameraFrame, EngineeringCameraFrame

    coord = SkyCoord(x=1 * u.m, y=2 * u.m, frame=CameraFrame())
    eng_coord = coord.transform_to(EngineeringCameraFrame())

    assert eng_coord.x == -coord.y
    assert eng_coord.y == -coord.x

    back = eng_coord.transform_to(CameraFrame())
    assert back.x == coord.x
    assert back.y == coord.y
