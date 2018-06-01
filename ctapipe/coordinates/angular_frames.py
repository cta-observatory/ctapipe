"""This module defines the important coordinate systems to be used in
reconstruction with the CTA pipeline and the transformations between
this different systems.

For examples on usage see examples/coordinate_transformations.py

This code is based on the coordinate transformations performed in the
read_hess code

TODO:

- Check cartesian system is still accurate for the nominal and
  telescope systems (may need a spherical system)
- Benchmark transformation time
"""

import astropy.units as u
import numpy as np
from numpy import cos, sin

from ctapipe.coordinates.coordinate_base import *
from ctapipe.coordinates.utils import *

__all__ = [
    'CameraFrame',
    'TelescopeFrame',
    'NominalFrame',
    'HorizonFrame'
]


# Transformation between nominal and AltAz system

def nominal_to_horizon(norm_coord):
    """
    Transformation from nominal system to astropy AltAz system

    Parameters
    ----------
    norm_coord: `astropy.coordinates.SkyCoord`
        nominal system

    Returns
    -------
    AltAz Coordinates
    """
    alt_norm, az_norm = norm_coord.array_pointing.alt.to(u.rad).value, \
                        norm_coord.array_pointing.az.to(u.rad).value

    if type(norm_coord.x.value).__module__ != np.__name__:
        x_off = np.zeros(1)
        x_off[0] = norm_coord.x.to(u.rad).value
        y_off = np.zeros(1)
        y_off[0] = norm_coord.y.to(u.rad).value
    else:
        x_off = norm_coord.x.to(u.rad).value
        y_off = norm_coord.y.to(u.rad).value

    altitude, azimuth = offset_to_horizon(x_off, y_off, az_norm, alt_norm)

    return HorizonFrame(altitude * u.rad, azimuth * u.rad)


def horizon_to_nominal(altaz_coord):
    """
    Transformation from astropy AltAz system to nominal system

    Parameters
    ----------
    altaz_coord: `astropy.coordinates.SkyCoord`
        AltAz system

    Returns
    -------
    nominal Coordinates
    """
    alt_norm, az_norm = altaz_coord.array_pointing.alt.to(u.rad).value, \
                        altaz_coord.array_pointing.az.to(u.rad).value
    azimuth = altaz_coord.az.to(u.rad).value
    altitude = altaz_coord.alt.to(u.rad).value

    x_off, y_off = horizon_to_offset(azimuth, altitude, az_norm, alt_norm)

    return NominalFrame(x_off * u.rad, y_off * u.rad, **altaz_coord.copy_properties())


# Transformation between telescope and nominal frames

def telescope_to_nominal(tel_coord):
    """
    Coordinate transformation from telescope frame to nominal frame

    Parameters
    ----------
    tel_coord: `astropy.coordinates.SkyCoord`
        TelescopeFrame system

    Returns
    -------
    NominalFrame coordinates
    """
    alt_tel, az_tel = tel_coord.telescope_pointing.alt.to(u.rad).value, \
                      tel_coord.telescope_pointing.az.to(u.rad).value
    alt_norm, az_norm = tel_coord.array_pointing.alt.to(u.rad).value, \
                        tel_coord.array_pointing.az.to(u.rad).value

    alt_trans, az_trans = offset_to_horizon(
        tel_coord.x.to(u.rad).value, tel_coord.y.to(u.rad).value, az_tel, alt_tel)

    x_off, y_off = horizon_to_offset(az_trans, alt_trans, az_norm, alt_norm)

    return NominalFrame(x_off * u.rad, y_off * u.rad, **tel_coord.copy_properties())


def nominal_to_telescope(norm_coord):
    """
    Coordinate transformation from nominal to telescope system

    Parameters
    ----------
    norm_coord: `astropy.coordinates.SkyCoord`
        NominalFrame system

    Returns
    -------
    TelescopeFrame coordinates

    """
    alt_tel, az_tel = norm_coord.telescope_pointing.alt.to(u.rad).value, \
                      norm_coord.telescope_pointing.az.to(u.rad).value
    alt_norm, az_norm = norm_coord.array_pointing.alt.to(u.rad).value, \
                        norm_coord.array_pointing.az.to(u.rad).value

    alt_trans, az_trans = offset_to_horizon(
        norm_coord.x.to(u.rad).value, norm_coord.y.to(u.rad).value, az_norm, alt_norm)
    x_off, y_off = horizon_to_offset(az_trans, alt_trans, az_tel, alt_tel)

    return TelescopeFrame(x_off * u.rad, y_off * u.rad, **norm_coord.copy_properties())


# Transformations between camera frame and telescope frame
def camera_to_telescope(camera_coord):
    """
    Transformation between CameraFrame and TelescopeFrame

    Parameters
    ----------
    camera_coord: `astropy.coordinates.SkyCoord`
        CameraFrame system
    Returns
    -------
    TelescopeFrame coordinate
    """
    x_pos = camera_coord.x.to(u.m).value
    y_pos = camera_coord.y.to(u.m).value

    rot = camera_coord.rotation.to(u.rad).value
    focal_length = camera_coord.focal_length.to(u.m).value

    if rot == 0:
        x_rotated = x_pos
        y_rotated = y_pos
    else:
        x_rotated = x_pos * cos(rot) - y_pos * sin(rot)
        y_rotated = x_pos * sin(rot) + y_pos * cos(rot)
    print(x_rotated)

    x_rotated = (x_rotated / focal_length)
    y_rotated = (y_rotated / focal_length)

    return TelescopeFrame(x_rotated * u.rad, y_rotated * u.rad,
                          **camera_coord.copy_properties())


def telescope_to_camera(telescope_coord):
    """
    Transformation between TelescopeFrame and CameraFrame

    Parameters
    ----------
    telescope_coord: `astropy.coordinates.SkyCoord`
        TelescopeFrame system
    Returns
    -------
    CameraFrame Coordinates
    """
    x_pos = telescope_coord.x.to(u.rad).value
    y_pos = telescope_coord.y.to(u.rad).value
    # reverse the rotation applied to get to this system
    rot = telescope_coord.rotation.to(u.rad).value * -1
    focal_length = telescope_coord.focal_length.to(u.m).value

    if rot == 0:  # if no rotation applied save a few cycles
        x_rotated = x_pos
        y_rotated = y_pos
    else:  # or else rotate all positions around the camera centre
        x_rotated = x_pos * cos(rot) - y_pos * sin(rot)
        y_rotated = x_pos * sin(rot) + y_pos * cos(rot)

    # Remove distance units here as we are using small angle approx
    x_rotated = x_rotated * focal_length
    y_rotated = y_rotated * focal_length

    return CameraFrame(x_rotated * u.m, y_rotated * u.m, **telescope_coord.copy_properties())


class AngularCoordinate(BaseCoordinate):
    """

    """
    system_order = np.array(["CameraFrame", "TelescopeFrame",
                             "NominalFrame", "HorizonFrame"])

    transformations = np.array([camera_to_telescope, telescope_to_nominal,
                                nominal_to_horizon])
    reverse_transformations = np.array([telescope_to_camera, nominal_to_telescope,
                                        horizon_to_nominal])

    def __init__(self, focal_length=None, telescope_pointing=None, array_pointing=None,
                 rotation=0 * u.deg):
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
        self.focal_length = focal_length
        self.telescope_pointing = telescope_pointing
        self.array_pointing = array_pointing
        self.rotation = rotation

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


class CameraFrame(AngularCoordinate, Cartesian2D):
    """Camera coordinate frame.  The camera frame is a simple physical
    cartesian frame, describing the 2 dimensional position of objects
    in the focal plane of the telescope Most Typically this will be
    used to describe the positions of the pixels in the focal plane
    """

    def __init__(self, x=None, y=None, **kwargs):
        super().__init__(**kwargs)

        self.x = x
        self.y = y


class TelescopeFrame(AngularCoordinate, Cartesian2D):
    """Telescope coordinate frame.  Cartesian system to describe the
    angular offset of a given position in reference to pointing
    direction of a given telescope When pointing corrections become
    available they should be applied to the transformation between
    this frame and the camera frame
    """

    def __init__(self, x=None, y=None, **kwargs):
        super().__init__(**kwargs)

        self.x = x
        self.y = y


class NominalFrame(AngularCoordinate, Cartesian2D):
    """Nominal coordinate frame.  Cartesian system to describe the angular
    offset of a given position in reference to pointing direction of a
    nominal array pointing position. In most cases this frame is the
    same as the telescope frame, however in the case of divergent
    pointing they will differ.  Event reconstruction should be
    performed in this system
    """

    def __init__(self, x=None, y=None, **kwargs):
        super().__init__(**kwargs)

        self.x = x
        self.y = y


class HorizonFrame(AngularCoordinate, UnitSpherical):
    """Horizon coordinate frame. Spherical system used to describe the direction
    of a given position, in terms of the altitude and azimuth of the system. In
    practice this is functionally identical as the astropy AltAz system, but this
    implementation allows us to pass array pointing information, allowing us to directly
    transform to the Horizon Frame from the Camera system.
    The Following attributes are carried over from the telescope frame
    to allow a direct transformation from the camera frame
   """

    def __init__(self, alt=None, az=None, **kwargs):
        super().__init__(**kwargs)

        self.alt = alt
        self.az = az
        self.phi = self.az
        self.theta = self.alt
