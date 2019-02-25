#!/usr/bin/env python3
"""
Some simple examples of using the frames and transformations
defined in `ctapipe.coordinates`

"""
import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np

from ctapipe.coordinates import (
    CameraFrame,
    TelescopeFrame,
    GroundFrame,
    TiltedGroundFrame,
    NominalFrame,
    AltAz,
    project_to_ground,
)


# Convert camera frame to telescope frame


def cam_to_tel():

    # Coordinates in any fram can be given as a numpy array of the xyz positions
    # e.g. in this case the position on pixels in the camera
    pix_x = np.ones(2048) * u.m
    pix_y = np.ones(2048) * u.m

    # first define the camera frame
    camera_frame = CameraFrame(focal_length=15 * u.m)
    # create a coordinate in that frame
    camera_coord = SkyCoord(pix_x, pix_y, frame=camera_frame)

    # then use transform to function to convert to a new system making sure
    # to give the required values for the conversion (these are not checked
    # yet)
    telescope_coord = camera_coord.transform_to(TelescopeFrame())

    # Print coordinates in the new frame
    print("Telescope Coordinate", telescope_coord)

    # Transforming back is then easy
    camera_coord2 = telescope_coord.transform_to(camera_frame)

    # We can easily check the distance between 2 coordinates in the same frame
    # In this case they should be the same
    print("Separation", np.sum(camera_coord.separation_3d(camera_coord2)))


# The astropy system is clever enough to transform through several intermediate
# steps to get to the sytem you want (provided it has sufficient information)
def cam_to_nom():
    pix_x = np.ones(2048) * u.m
    pix_y = np.ones(2048) * u.m

    pointing_direction = SkyCoord(alt=70 * u.deg, az=180 * u.deg, frame=AltAz())
    camera_frame = CameraFrame(
        focal_length=15 * u.m,
        telescope_pointing=pointing_direction
    )
    camera_coord = SkyCoord(pix_x, pix_y, frame=camera_frame)

    # In this case we bypass the telescope system
    nominal_frame = NominalFrame(
        origin=AltAz(alt=75 * u.deg, az=180 * u.deg)
    )
    nom_coord = camera_coord.transform_to(nominal_frame)

    horizon = camera_coord.transform_to(AltAz())

    print("Nominal Coordinate", nom_coord)
    print("Horizon coordinate", horizon)


# Once we are at the nominal system where most reconstruction will be done we
# can then convert to AltAz (currently we cannot transform directly from camera)
def nominal_to_altaz():

    nom = SkyCoord(
        x=0 * u.deg,
        y=0 * u.deg,
        frame=NominalFrame(origin=AltAz(alt=75 * u.deg, az=180 * u.deg))
    )
    alt_az = nom.transform_to(AltAz())
    print("HorizonCoordinate", alt_az)
    # Provided we know when and where the AltAz was measured we can them
    # convert this to any astronomical


# We also have the ground and tilted ground systems needed for core
# reconstruction
def grd_to_tilt():
    grd_coord = GroundFrame(x=1 * u.m, y=2 * u.m, z=0 * u.m)
    tilt_coord = grd_coord.transform_to(
        TiltedGroundFrame(
            pointing_direction=AltAz(alt=90 * u.deg, az=180 * u.deg)
        )
    )
    print(project_to_ground(tilt_coord))
    print("Tilted Coordinate", tilt_coord)


if __name__ == '__main__':
    cam_to_tel()
    cam_to_nom()
    nominal_to_altaz()
    grd_to_tilt()
