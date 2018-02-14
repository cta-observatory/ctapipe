#!/usr/bin/env python3
"""
Some simple examples of using the frames and transformations
defined in `ctapipe.coordinates`

"""
import astropy.units as u
import numpy as np

from ctapipe.coordinates import CameraFrame, TelescopeFrame, GroundFrame, \
    TiltedGroundFrame, NominalFrame, HorizonFrame, project_to_ground


# Convert camera frame to telescope frame


def cam_to_tel():

    # Coordinates in any fram can be given as a numpy array of the xyz positions
    # e.g. in this case the position on pixels in the camera
    pix = [np.ones(2048), np.ones(2048), np.zeros(2048)] * u.m
    # first define the camera frame
    camera_coord = CameraFrame(pix, focal_length=15 * u.m, rotation=0 * u.deg)

    # then use transform to function to convert to a new system making sure
    # to give the required values for the conversion (these are not checked
    # yet)
    telescope_coord = camera_coord.transform_to(TelescopeFrame())

    # Print coordinates in the new frame
    print("Telescope Coordinate", telescope_coord)

    # Transforming back is then easy
    camera_coord2 = telescope_coord.transform_to(
        CameraFrame(focal_length=15 * u.m, rotation=0 * u.deg)
    )

    # We can easily check the distance between 2 coordinates in the same frame
    # In this case they should be the same
    print("Separation", np.sum(camera_coord.separation_3d(camera_coord2)))


# The astropy system is clever enough to transform through several intermediate
# steps to get to the sytem you want (provided it has sufficient information)
def cam_to_nom():
    pix = [np.ones(2048), np.ones(2048), np.zeros(2048)] * u.m
    camera_coord = CameraFrame(pix, focal_length=15 * u.m)
    # In this case we bypass the telescope system
    nom_coord = camera_coord.transform_to(
        NominalFrame(
            pointing_direction=HorizonFrame(alt=70 * u.deg, az=180 * u.deg),
            array_direction=HorizonFrame(alt=75 * u.deg, az=180 * u.deg)
        )
    )
    alt_az = camera_coord.transform_to(
        HorizonFrame(
            pointing_direction=HorizonFrame(alt=70 * u.deg, az=180 * u.deg),
            array_direction=HorizonFrame(alt=75 * u.deg, az=180 * u.deg)
        )
    )

    print("Nominal Coordinate", nom_coord)
    print("AltAz coordinate", alt_az)


# Once we are at the nominal system where most reconstruction will be done we
# can then convert to AltAz (currently we cannot transform directly from camera)
def nominal_to_altaz():
    t = np.zeros(10)
    t[5] = 1
    nom = NominalFrame(
        x=t * u.deg,
        y=t * u.deg,
        array_direction=HorizonFrame(alt=75 * u.deg, az=180 * u.deg)
    )
    alt_az = nom.transform_to(HorizonFrame)
    print("AltAz Coordinate", alt_az)
    # Provided we know when and where the AltAz was measured we can them
    # convert this to any astronomical


# We also have the ground and tilted ground systems needed for core
# reconstruction
def grd_to_tilt():
    grd_coord = GroundFrame(x=1 * u.m, y=2 * u.m, z=0 * u.m)
    tilt_coord = grd_coord.transform_to(
        TiltedGroundFrame(
            pointing_direction=HorizonFrame(alt=90 * u.deg, az=180 * u.deg)
        )
    )
    print(project_to_ground(tilt_coord))
    print("Tilted Coordinate", tilt_coord)


if __name__ == '__main__':
    cam_to_tel()
    cam_to_nom()
    nominal_to_altaz()
    grd_to_tilt()
