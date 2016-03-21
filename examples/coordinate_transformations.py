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
from astropy.coordinates import SkyCoord, Angle, AltAz
#from astropy.coordinates.builtin_frames.altaz import AltAz
import timeit

from ctapipe.coordinates import CameraFrame, TelescopeFrame, GroundFrame, TiltedGroundFrame, NominalFrame
from astropy import coordinates as c

import math
import numpy as np

def cam_to_tel():

    pix = [np.ones(2048),np.ones(2048),np.zeros(2048)] * u.m
    camera_coord = CameraFrame(pix)
    telescope_coord = camera_coord.transform_to(TelescopeFrame(focal_length = 10*u.m,rotation=0*u.deg))

def tel_to_cam():
    pix = [np.ones(2048),np.ones(2048),np.zeros(2048)] * u.deg
    telescope_coord = TelescopeFrame(pix,focal_length = 10*u.m)
    camera_coord = telescope_coord.transform_to(CameraFrame())

def cam_to_nom():
    pix = [np.ones(2048),np.ones(2048),np.zeros(2048)] * u.m
    camera_coord = CameraFrame(pix)


def test_camera_telescope_transform():
    camera_coord = CameraFrame(x=1*u.m, y=2*u.m, z=0*u.m)
    pix = [[1,1,1],[2,2,2],[3,3,3]] * u.m
    #print (pix)
    camera_coord = CameraFrame(pix)
    #print(camera_coord,pix)

    telescope_coord = camera_coord.transform_to(TelescopeFrame(focal_length = 10*u.m))
    #print(telescope_coord)

    camera_coord2 = telescope_coord.transform_to(CameraFrame)
    #print(camera_coord2)

    grd_coord = GroundFrame(x=1*u.m, y=2*u.m, z=0*u.m)
    #print (grd_coord)
    alt = 50 * u.deg
    #print(alt.to(u.rad)/u.rad)
    tilt_coord = grd_coord.transform_to(TiltedGroundFrame(pointing_direction = [90*u.deg,180*u.deg]))
    grd_coord2 = tilt_coord.transform_to(GroundFrame())

    #print (tilt_coord)
    #print (grd_coord2)
    tel_coord = TelescopeFrame(0.1*u.deg,0.1*u.deg,0.0*u.deg,pointing_direction = [70*u.deg,180*u.deg])
    nominal_coord = tel_coord.transform_to(NominalFrame(pointing_direction = [75*u.deg,180*u.deg]))
    #print(tel_coord,nominal_coord)
    #print(nominal_coord.transform_to(TelescopeFrame(pointing_direction=[70*u.deg,180*u.deg])))
    obs = Time( "2015-10-11 15:17:45.3", format="iso", scale="utc")
    paris = c.EarthLocation( lat=48.8567*u.deg, lon=2.3508*u.deg )
    a = AltAz(1*u.deg, 2*u.deg)
    a2 = AltAz()

    print(a,a2)
    #altaz = SkyCoord(0,0,frame='altaz')
    #print(nominal_coord.transform_to(a))
    print("here")

if __name__ == '__main__':
    #test_camera_telescope_transform()
    print (timeit.timeit(cam_to_tel, number=1000)/1000.)
    print (timeit.timeit(tel_to_cam, number=1000)/1000.)