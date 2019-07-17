'''
Tests for the conversion between camera coordinate frames
'''
from astropy.coordinates import SkyCoord
import astropy.units as u


def test_conversion():
    '''
    Test conversion between CameraFrame and EngineeringCameraFrame
    '''
    from ctapipe.coordinates import CameraFrame, EngineeringCameraFrame

    coords = SkyCoord(x=[3, 1] * u.m, y=[2, 4] * u.m, frame=CameraFrame())

    for coord in coords:
        eng_coord = coord.transform_to(EngineeringCameraFrame())

        assert eng_coord.x == -coord.y
        assert eng_coord.y == -coord.x

        back = eng_coord.transform_to(CameraFrame())
        assert back.x == coord.x
        assert back.y == coord.y

        eng_coord = coord.transform_to(EngineeringCameraFrame(n_mirrors=2))

        assert eng_coord.x == coord.y
        assert eng_coord.y == -coord.x

        back = eng_coord.transform_to(CameraFrame())
        assert back.x == coord.x
        assert back.y == coord.y
