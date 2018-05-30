"""This module defines the important coordinate systems to be used in
reconstruction with the CTA pipeline and the transformations between
this different systems. Frames and transformations are defined using
the astropy.coordinates framework. This module defines transformations
for ground based cartesian and planar systems.

For examples on usage see examples/coordinate_transformations.py

This code is based on the coordinate transformations performed in the
read_hess code

TODO:

- Tests Tests Tests!
"""

import astropy.units as u
import numpy as np

from ctapipe.coordinates.coordinate_base import *
from ctapipe.coordinates.utils import *

__all__ = [
    'GroundFrame',
    'TiltedGroundFrame',
    'project_to_ground'
]


def ground_to_tilted(ground_coord):
    """
    Transformation from ground system to tilted ground system

    Parameters
    ----------
    ground_coord: `astropy.coordinates.SkyCoord`
        GroundFrame system
    tilted_coord: `astropy.coordinates.SkyCoord`
        TiltedGroundFrame system

    Returns
    -------
    TiltedGroundFrame coordinates
    """
    x_grd = ground_coord.x
    y_grd = ground_coord.y
    z_grd = ground_coord.z

    altitude, azimuth = ground_coord.pointing_direction.alt, \
                        ground_coord.pointing_direction.az
    altitude = altitude.to(u.rad)
    azimuth = azimuth.to(u.rad)
    trans = get_shower_trans_matrix(azimuth, altitude)

    x_tilt = trans[0][0] * x_grd + trans[0][1] * y_grd + trans[0][2] * z_grd
    y_tilt = trans[1][0] * x_grd + trans[1][1] * y_grd + trans[1][2] * z_grd
    return TiltedGroundFrame(x_tilt, y_tilt, **ground_coord.copy_properties())


def tilted_to_ground(tilted_coord):
    """
    Transformation from tilted ground system to  ground system

    Parameters
    ----------
    tilted_coord: `astropy.coordinates.SkyCoord`
        TiltedGroundFrame system
    ground_coord: `astropy.coordinates.SkyCoord`
        GroundFrame system

    Returns
    -------
    GroundFrame coordinates
    """
    x_tilt = tilted_coord.x
    y_tilt = tilted_coord.y

    altitude, azimuth = tilted_coord.pointing_direction.alt, \
                        tilted_coord.pointing_direction.az
    altitude = altitude.to(u.rad)
    azimuth = azimuth.to(u.rad)

    trans = get_shower_trans_matrix(azimuth, altitude)

    x_grd = trans[0][0] * x_tilt + trans[1][0] * y_tilt
    y_grd = trans[0][1] * x_tilt + trans[1][1] * y_tilt
    z_grd = trans[0][2] * x_tilt + trans[1][2] * y_tilt

    return GroundFrame(x_grd, y_grd, z_grd, **tilted_coord.copy_properties())


def project_to_ground(tilt_system):
    """Project position in the tilted system onto the ground. This is
    needed as the standard transformation will return the 3d position
    of the tilted frame. This projection may ultimately be the
    standard use case so may be implemented in the tilted to ground
    transformation

    Parameters
    ----------
    tilt_system: `astropy.coordinates.SkyCoord`
        coorinate in the the tilted ground system

    Returns
    -------
    Projection of tilted system onto the ground (GroundSystem)

    """
    ground_system = tilt_system.transform_to(GroundFrame)

    unit = ground_system.x.unit
    x_initial = ground_system.x.value
    y_initial = ground_system.y.value
    z_initial = ground_system.z.value

    trans = get_shower_trans_matrix(tilt_system.pointing_direction.az,
                                    tilt_system.pointing_direction.alt)

    x_projected = x_initial - trans[2][0] * z_initial / trans[2][2]
    y_projected = y_initial - trans[2][1] * z_initial / trans[2][2]

    return GroundFrame(x=x_projected * unit, y=y_projected * unit, z=0 * unit)


class GroundCoordinate(BaseCoordinate):
    """

    """
    system_order = np.array(["GroundFrame", "TiltedGroundFrame"])

    transformations = np.array([ground_to_tilted])
    reverse_transformations = np.array([project_to_ground])

    def __init__(self, pointing_direction=None):
        """
        Parameters
        ----------
        focal_length: ndarray
            Focal length of telescope
        telescope_pointing: HorizonFrame
            Pointing direction of telescope
        array_pointing: HorizonFrame
            Pointing direction of array
        rotation: ndarray
            Rotation angle of camera in telescope
        """
        self.pointing_direction = pointing_direction

        prop_dict = dict()
        for key in self.__dict__:
            prop_dict[key] = self.__dict__[key]
        self.properties = prop_dict

        return

    def copy_properties(self):
        """
        Create a copy of the shared class parameters to share with other classes

        Returns
        -------
        dict: Dictionary of shared class parameters
        """
        properties = self.properties
        return properties


class GroundFrame(GroundCoordinate, Cartesian3D):
    """Ground coordinate frame.  The ground coordinate frame is a simple
    cartesian frame describing the 3 dimensional position of objects
    compared to the array ground level in relation to the nomial
    centre of the array.  Typically this frame will be used for
    describing the position on telescopes and equipment
    """

    def __init__(self, x=None, y=None, z=None, **kwargs):
        super().__init__(**kwargs)

        self.x = x
        self.y = y
        self.z = z

        return


class TiltedGroundFrame(GroundCoordinate, Cartesian2D):
    """Tilted ground coordinate frame.  The tilted ground coordinate frame
    is a cartesian system describing the 2 dimensional projected
    positions of objects in a tilted plane described by
    pointing_direction Typically this frame will be used for the
    reconstruction of the shower core position
    """

    def __init__(self, x=None, y=None, **kwargs):
        super().__init__(**kwargs)

        self.x = x
        self.y = y

        return
