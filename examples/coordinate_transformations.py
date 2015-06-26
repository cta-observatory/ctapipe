"""
Some simple examples of using the frames and transformations
defined in `ctapipe.coordinates`

Basically this is me learning how `astropy.coordinates` works,
and how the coordinate frames and transformations work at the same time
... it could be all wrong and comments / contributions are welcome!

TODO: Goal: Define CameraFrame (attributes: pointing Alt, Az + focal length) and connect it to AltAzFrame
"""
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, Angle
from ctapipe.coordinates import CameraFrame, TelescopeFrame

def test_camera_telescope_transform():
    camera_coord = CameraFrame(x=1*u.m, y=2*u.m, z=0*u.m)
    print(camera_coord)

    telescope_coord = camera_coord.transform_to(TelescopeFrame)
    print(telescope_coord)

    camera_coord2 = telescope_coord.transform_to(CameraFrame)
    print(camera_coord2)


if __name__ == '__main__':
    test_camera_telescope_transform()
