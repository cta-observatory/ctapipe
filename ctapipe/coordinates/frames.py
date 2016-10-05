"""This module defines the important coordinate systems to be used in
reconstruction with the CTA pipeline and the transformations between
this different systems. Frames and transformations are defined using
the astropy.coordinates framework.

For examples on usage see examples/coordinate_transformations.py

This code is based on the coordinate transformations performed in the
read_hess code

TODO:

- Tests Tests Tests!
- Check cartesian system is still accurate for the nominal and
  telescope systems (may need a spherical system)
- Benchmark transformation times
- should use `astropy.coordinates.Angle` for all angles here 

"""

import numpy as np
import astropy.units as u
from astropy.coordinates import (BaseCoordinateFrame, FrameAttribute,
                                 SphericalRepresentation,
                                 CartesianRepresentation,
                                 RepresentationMapping,
                                 FunctionTransform, SkyCoord)
from astropy.coordinates import AltAz
from astropy.coordinates import frame_transform_graph
from numpy import cos, sin, arctan, arctan2, arcsin, sqrt, arccos, tan

__all__ = [
    'CameraFrame',
    'TelescopeFrame',
    'GroundFrame',
    'TiltedGroundFrame',
    'NominalFrame',
    'project_to_ground'
]


class CameraFrame(BaseCoordinateFrame):
    """Camera coordinate frame.  The camera frame is a simple physical
    cartesian frame, describing the 2 dimensional position of objects
    in the focal plane of the telescope Most Typically this will be
    used to describe the positions of the pixels in the focal plane

    Frame attributes:  None

    """
    default_representation = CartesianRepresentation


class TelescopeFrame(BaseCoordinateFrame):
    """Telescope coordinate frame.  Cartesian system to describe the
    angular offset of a given position in reference to pointing
    direction of a given telescope When pointing corrections become
    available they should be applied to the transformation between
    this frame and the camera frame

    Frame attributes:

    * ``focal_length``
        Focal length of the telescope as a unit quantity (usually meters)
    * ``rotation``
        Rotation angle of the camera (0 deg in most cases) 
    * ``pointing_direction``
        Alt,Az direction of the telescope pointing

    """
    default_representation = CartesianRepresentation

    focal_length = FrameAttribute(default=None)  # focal_length
    rotation = FrameAttribute(default=0 * u.deg)
    pointing_direction = FrameAttribute(default=None)


class NominalFrame(BaseCoordinateFrame):
    """Nominal coordinate frame.  Cartesian system to describe the angular
    offset of a given position in reference to pointing direction of a
    nominal array pointing position. In most cases this frame is the
    same as the telescope frame, however in the case of divergent
    pointing they will differ.  Event reconstruction should be
    performed in this system

    Frame attributes:

    * ``pointing_direction``
      Alt,Az direction of the array pointing

    The Following attributes are carried over from the telescope frame
    to allow a direct transformation from the camera frame

    * ``focal_length``
      Focal length of the telescope 
    * ``rotation``
      Rotation angle of the camera (0 in most cases) [deg]
    * ``pointing_direction``
      Alt,Az direction of the telescope pointing

    """
    default_representation = CartesianRepresentation
    array_direction = FrameAttribute(default=None)
    pointing_direction = FrameAttribute(default=None)

    rotation = FrameAttribute(default=0 * u.deg)
    focal_length = FrameAttribute(default=None)


def altaz_to_offset(obj_azimuth, obj_altitude, azimuth, altitude):
    """
    Function to convert a given altitude and azimuth to a cartesian angular
    angular offset with regard to a give reference system
    (This function is directly lifted from read_hess)

    Parameters
    ----------
    obj_azimuth: float
        Event azimuth (radians)
    obj_altitude: float
        Event altitude (radians)
    azimuth: float
        Reference system azimuth (radians)
    altitude: float
        Reference system altitude (radians)

    Returns
    -------
    xoff,yoff: (float,float)
        Offset of the event in the reference system (in radians)
    """

    daz = obj_azimuth - azimuth
    coa = cos(obj_altitude)

    xp0 = -cos(daz) * coa
    yp0 = sin(daz) * coa
    zp0 = sin(obj_altitude)

    cx = sin(altitude)
    sx = cos(altitude)

    xp1 = cx * xp0 + sx * zp0
    yp1 = yp0
    zp1 = -sx * xp0 + cx * zp0

    q = arccos(zp1)
    d = tan(q)
    alpha = arctan2(yp1, xp1)

    xoff = d * cos(alpha)
    yoff = d * sin(alpha)

    return xoff, yoff


def offset_to_altaz(xoff, yoff, azimuth, altitude):
    """Function to convert an angular offset with regard to a give
    reference system to an an absolute altitude and azimuth (This
    function is directly lifted from read_hess)

    Parameters
    ----------
    xoff: float
        X offset of the event in the reference system
    yoff: float
        Y offset of the event in the reference system
    azimuth: float
        Reference system azimuth (radians)
    altitude: float
        Reference system altitude (radians)

    Returns
    -------
    obj_altitude,obj_azimuth: (float,float)
        Absolute altitude and azimuth of the event
    """

    unit = azimuth.unit

    xoff = xoff.to(u.rad).value
    yoff = yoff.to(u.rad).value
    azimuth = azimuth.to(u.rad).value
    altitude = altitude.to(u.rad).value

    d = sqrt(xoff * xoff + yoff * yoff)
    pos = np.where(d == 0)  # find offset 0 positions
    if len(pos[0]) > 0:
        d[pos] = 1e-12  # add a very small offset to prevent math errors

    q = arctan(d)

    sq = sin(q)
    xp1 = xoff * (sq / d)
    yp1 = yoff * (sq / d)
    zp1 = cos(q)

    cx = sin(altitude)
    sx = cos(altitude)

    xp0 = cx * xp1 - sx * zp1
    yp0 = yp1
    zp0 = sx * xp1 + cx * zp1
    obj_altitude = arcsin(zp0)
    obj_azimuth = arctan2(yp0, -xp0) + azimuth

    if len(pos[0]) > 0:
        obj_altitude[pos] = altitude
        obj_azimuth[pos] = azimuth

    obj_altitude = obj_altitude * u.rad
    obj_azimuth = obj_azimuth * u.rad

    # if obj_azimuth.value < 0.:
    #    obj_azimuth += 2.*pi
    # elif obj_azimuth.value >= (2.*pi ):
    #    obj_azimuth -= 2.*pi

    return obj_altitude.to(unit), obj_azimuth.to(unit)

# Transformation between nominal and AltAz system


@frame_transform_graph.transform(FunctionTransform, NominalFrame, AltAz)
def nominal_to_altaz(norm_coord, altaz_coord):
    """
    Transformation from nominal system to astropy AltAz system

    Parameters
    ----------
    norm_coord: `astropy.coordinates.SkyCoord`
        nominal system
    altaz_coord: `astropy.coordinates.SkyCoord`
        AltAz system

    Returns
    -------
    AltAz Coordinates
    """
    alt_norm, az_norm = norm_coord.array_direction

    if type(norm_coord.x.value).__module__ != np.__name__:
        x = np.zeros(1)
        x[0] = norm_coord.x.value
        x = x * norm_coord.x.unit
        y = np.zeros(1)
        y[0] = norm_coord.y.value
        y = y * norm_coord.y.unit
    else:
        x = norm_coord.x
        y = norm_coord.y

    alt, az = offset_to_altaz(x, y, az_norm, alt_norm)
    altaz_coord = AltAz(az=az.to(u.deg), alt=alt.to(u.deg))

    return altaz_coord


@frame_transform_graph.transform(FunctionTransform, AltAz, NominalFrame)
def nominal_to_altaz(altaz_coord, norm_coord):
    """
    Transformation from astropy AltAz system to nominal system

    Parameters
    ----------
    altaz_coord: `astropy.coordinates.SkyCoord`
        AltAz system
    norm_coord: `astropy.coordinates.SkyCoord`
        nominal system

    Returns
    -------
    nominal Coordinates
    """
    alt_norm, az_norm = norm_coord.array_direction
    az = altaz_coord.az
    alt = altaz_coord.alt
    x, y = altaz_to_offset(az, alt, az_norm, alt_norm)
    x = x * u.rad
    y = y * u.rad
    representation = CartesianRepresentation(
        x.to(u.deg), y.to(u.deg), 0 * u.deg)

    return norm_coord.realize_frame(representation)

# Transformation between telescope and nominal frames


@frame_transform_graph.transform(FunctionTransform, TelescopeFrame, NominalFrame)
def telescope_to_nominal(tel_coord, norm_frame):
    """
    Coordinate transformation from telescope frame to nominal frame

    Parameters
    ----------
    tel_coord: `astropy.coordinates.SkyCoord`
        TelescopeFrame system
    norm_frame: `astropy.coordinates.SkyCoord`
        NominalFrame system

    Returns
    -------
    NominalFrame coordinates
    """
    alt_tel, az_tel = tel_coord.pointing_direction
    alt_norm, az_norm = norm_frame.array_direction
    alt_trans, az_trans = offset_to_altaz(
        tel_coord.x, tel_coord.y, az_tel, alt_tel)

    x, y = altaz_to_offset(az_trans, alt_trans, az_norm, alt_norm)
    x = x * u.rad
    y = y * u.rad

    representation = CartesianRepresentation(
        x.to(tel_coord.x.unit), y.to(tel_coord.x.unit), 0 * tel_coord.x.unit)

    return norm_frame.realize_frame(representation)


@frame_transform_graph.transform(FunctionTransform, NominalFrame, TelescopeFrame)
def nominal_to_telescope(norm_coord, tel_frame):
    """
    Coordinate transformation from nominal to telescope system

    Parameters
    ----------
    norm_coord: `astropy.coordinates.SkyCoord`
        NominalFrame system
    tel_frame: `astropy.coordinates.SkyCoord`
        TelescopeFrame system

    Returns
    -------
    TelescopeFrame coordinates

    """
    alt_tel, az_tel = tel_frame.pointing_direction
    alt_norm, az_norm = norm_coord.array_direction

    alt_trans, az_trans = offset_to_altaz(
        norm_coord.x, norm_coord.y, az_norm, alt_norm)
    x, y = altaz_to_offset(az_trans, alt_trans, az_tel, alt_tel)
    x = x * u.rad
    y = y * u.rad

    representation = CartesianRepresentation(x.to(norm_coord.x.unit),
                                             y.to(norm_coord.x.unit),
                                             0 * norm_coord.x.unit)

    return tel_frame.realize_frame(representation)


# Transformations between camera frame and telescope frame
@frame_transform_graph.transform(FunctionTransform, CameraFrame, TelescopeFrame)
def camera_to_telescope(camera_coord, telescope_frame):
    """
    Transformation between CameraFrame and TelescopeFrame

    Parameters
    ----------
    camera_coord: `astropy.coordinates.SkyCoord`
        CameraFrame system
    telescope_frame: `astropy.coordinates.SkyCoord`
        TelescopeFrame system
    Returns
    -------
    TelescopeFrame coordinate
    """
    x_pos = camera_coord.cartesian.x
    y_pos = camera_coord.cartesian.y

    rot = telescope_frame.rotation
    if rot == 0:
        x = x_pos
        y = y_pos
    else:
        x = x_pos * cos(rot) - y_pos * sin(rot)
        y = x_pos * sin(rot) + y_pos * cos(rot)

    f = telescope_frame.focal_length

    x = (x / f) * u.rad
    y = (y / f) * u.rad
    representation = CartesianRepresentation(x, y, 0 * u.rad)

    return telescope_frame.realize_frame(representation)


@frame_transform_graph.transform(FunctionTransform, TelescopeFrame, CameraFrame)
def telescope_to_camera(telescope_coord, camera_frame):
    """
    Transformation between TelescopeFrame and CameraFrame

    Parameters
    ----------
    telescope_coord: `astropy.coordinates.SkyCoord`
        TelescopeFrame system
    camera_frame: `astropy.coordinates.SkyCoord`
        CameraFrame system

    Returns
    -------
    CameraFrame Coordinates
    """
    x_pos = telescope_coord.cartesian.x
    y_pos = telescope_coord.cartesian.y
    # reverse the rotation applied to get to this system
    rot = telescope_coord.rotation * -1

    if rot == 0:  # if no rotation applied save a few cycles
        x = x_pos
        y = y_pos
    else:  # or else rotate all positions around the camera centre
        x = x_pos * cos(rot) - y_pos * sin(rot)
        y = x_pos * sin(rot) + y_pos * cos(rot)

    f = telescope_coord.focal_length
    # Remove distance units here as we are using small angle approx
    x = x.to(u.rad) * (f / u.m)
    y = y.to(u.rad) * (f / u.m)

    representation = CartesianRepresentation(
        x.value * u.m, y.value * u.m, 0 * u.m)

    return camera_frame.realize_frame(representation)


# ############## Ground and Tilted system #####################

class GroundFrame(BaseCoordinateFrame):
    """Ground coordinate frame.  The ground coordinate frame is a simple
    cartesian frame describing the 3 dimensional position of objects
    compared to the array ground level in relation to the nomial
    centre of the array.  Typically this frame will be used for
    describing the position on telescopes and equipment

    Frame attributes: None

    """
    default_representation = CartesianRepresentation
    # Pointing direction of the tilted system (alt,az),
    # could be the telescope pointing direction or the reconstructed shower
    # direction
    pointing_direction = FrameAttribute(default=None)


class TiltedGroundFrame(BaseCoordinateFrame):
    """Tilted ground coordinate frame.  The tilted ground coordinate frame
    is a cartesian system describing the 2 dimensional projected
    positions of objects in a tilted plane described by
    pointing_direction Typically this frame will be used for the
    reconstruction of the shower core position

    Frame attributes:
    
    * ``pointing_direction`` 
      Alt,Az direction of the tilted reference plane

    """
    default_representation = CartesianRepresentation
    # Pointing direction of the tilted system (alt,az),
    # could be the telescope pointing direction or the reconstructed shower
    # direction
    pointing_direction = FrameAttribute(default=None)


def get_shower_trans_matrix(azimuth, altitude):
    """Get Transformation matrix for conversion from the ground system to
    the Tilted system and back again (This function is directly lifted
    from read_hess, probably could be streamlined using python
    functionality)

    Parameters
    ----------
    azimuth: float
        Azimuth angle of the tilted system used
    altitude: float
        Altitude angle of the tilted system used

    Returns
    -------
    trans: 3x3 ndarray transformation matrix
    """

    cos_z = sin(altitude)
    sin_z = cos(altitude)
    cos_az = cos(azimuth)
    sin_az = sin(azimuth)

    trans = np.zeros([3, 3])
    trans[0][0] = cos_z * cos_az
    trans[1][0] = sin_az
    trans[2][0] = sin_z * cos_az

    trans[0][1] = -cos_z * sin_az
    trans[1][1] = cos_az
    trans[2][1] = -sin_z * sin_az

    trans[0][2] = -sin_z
    trans[1][2] = 0.
    trans[2][2] = cos_z

    return trans


@frame_transform_graph.transform(FunctionTransform, GroundFrame,
                                 TiltedGroundFrame)
def ground_to_tilted(ground_coord, tilted_coord):
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
    x_grd = ground_coord.cartesian.x
    y_grd = ground_coord.cartesian.y
    z_grd = ground_coord.cartesian.z

    alt, az = tilted_coord.pointing_direction
    alt = alt.to(u.rad)
    az = az.to(u.rad)
    trans = get_shower_trans_matrix(az, alt)

    x_tilt = trans[0][0] * x_grd + trans[0][1] * y_grd + trans[0][2] * z_grd
    y_tilt = trans[1][0] * x_grd + trans[1][1] * y_grd + trans[1][2] * z_grd
    z_tilt = 0.0 * u.m

    representation = CartesianRepresentation(x_tilt, y_tilt, z_tilt)

    return tilted_coord.realize_frame(representation)


@frame_transform_graph.transform(FunctionTransform, TiltedGroundFrame,
                                 GroundFrame)
def tilted_to_ground(tilted_coord, ground_coord):
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
    x_tilt = tilted_coord.cartesian.x
    y_tilt = tilted_coord.cartesian.y

    alt, az = tilted_coord.pointing_direction
    alt = alt.to(u.rad)
    az = az.to(u.rad)

    trans = get_shower_trans_matrix(az, alt)

    x_grd = trans[0][0] * x_tilt + trans[1][0] * y_tilt
    y_grd = trans[0][1] * x_tilt + trans[1][1] * y_tilt
    z_grd = trans[0][2] * x_tilt + trans[1][2] * y_tilt

    representation = CartesianRepresentation(x_grd, y_grd, z_grd)

    grd = ground_coord.realize_frame(representation)
    return grd


def project_to_ground(tilt_system):
    """Project position in the tilted system onto the ground. This is
    needed as the standard transformation will return the 3d position
    of the tilted frame. This projection may untimately be the
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
    xh = ground_system.x.value
    yh = ground_system.y.value
    zh = ground_system.z.value

    trans = get_shower_trans_matrix(tilt_system.pointing_direction[
                                    1], tilt_system.pointing_direction[0])

    xc = xh - trans[2][0] * zh / trans[2][2]
    yc = yh - trans[2][1] * zh / trans[2][2]

    return GroundFrame(x=xc * unit, y=yc * unit, z=0 * unit)
