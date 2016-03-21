import numpy as np
import astropy.units as u
from astropy.coordinates import (BaseCoordinateFrame, FrameAttribute, SphericalRepresentation,
                                 CartesianRepresentation, RepresentationMapping, FunctionTransform,SkyCoord)
from astropy.coordinates.builtin_frames.altaz import AltAz
from astropy.coordinates import frame_transform_graph
from astropy.coordinates.angles import rotation_matrix
from numpy import cos,sin,arctan,arctan2,arcsin,sqrt,arccos,tan
from math import pi

__all__ = [
    'CameraFrame',
    'TelescopeFrame',
    'GroundFrame',
    'TiltedGroundFrame',
    'NominalFrame'
]


class CameraFrame(BaseCoordinateFrame):
    """Camera coordinate frame.
    """
    default_representation = CartesianRepresentation


class TelescopeFrame(BaseCoordinateFrame):
    """Telescope coordinate frame.

    Frame attributes:

    - Focal length
    """
    default_representation = CartesianRepresentation

    focal_length = FrameAttribute(default=None)
    rotation = FrameAttribute(default=0*u.deg)

    pointing_direction = FrameAttribute(default=None)

class NominalFrame(BaseCoordinateFrame):

    default_representation = CartesianRepresentation
    pointing_direction = FrameAttribute(default=None)


def altaz_to_offset (obj_azimuth,obj_altitude,azimuth,altitude):

    daz = obj_azimuth - azimuth
    coa = cos(obj_altitude)

    xp0 = -cos(daz) * coa
    yp0 = sin(daz) * coa
    zp0 = sin(obj_altitude)

    cx = sin(altitude)
    sx = cos(altitude)

    xp1 = cx*xp0 + sx*zp0
    yp1 = yp0
    zp1 = -sx*xp0 + cx*zp0

    if ( xp1 == 0 and yp1 == 0 ): # /* On-axis ? */
        return 0,0

    q = arccos(zp1)
    d = tan(q)
    alpha = arctan2(yp1,xp1)

    xoff = d * cos(alpha)
    yoff = d * sin(alpha)

    return xoff,yoff



def offset_to_altaz(xoff,yoff,azimuth,altitude):

    if  xoff == 0. and yoff == 0. : #/* Avoid division by zero */
        return altitude,azimuth
    else:
        d = sqrt(xoff*xoff+yoff*yoff)
        q = arctan(d.to(u.rad).value)

        sq = sin(q)
        xp1 = xoff * (sq/d)
        yp1 = yoff * (sq/d)
        zp1 = cos(q)

        cx = sin(altitude)
        sx = cos(altitude)

        xp0 = cx*xp1 - sx*zp1
        yp0 = yp1
        zp0 = sx*xp1 + cx*zp1

        obj_altitude = arcsin(zp0)
        obj_azimuth  = arctan2(yp0,-xp0) + azimuth
        if obj_azimuth.value < 0.:
            obj_azimuth += 2.*pi
        elif obj_azimuth.value >= (2.*pi ):
            obj_azimuth -= 2.*pi
        print (obj_altitude,obj_azimuth)

        return obj_altitude,obj_azimuth



@frame_transform_graph.transform(FunctionTransform, NominalFrame, AltAz)
def nominal_to_altaz(norm_coord,altaz_coord):

    alt_norm,az_norm = norm_coord.pointing_direction

    alt,az = offset_to_altaz(norm_coord.x,norm_coord.y,az_norm,alt_norm)
    print(alt.to(u.deg),az.to(u.deg))

    representation = SkyCoord(az.to(u.deg),alt.to(u.deg),frame='altaz')
    print("here")

    return altaz_coord.realize_frame(representation)

#############################################################

@frame_transform_graph.transform(FunctionTransform, TelescopeFrame, NominalFrame)
def telescope_to_nominal(tel_coord,norm_coord):

    alt_tel,az_tel = tel_coord.pointing_direction
    alt_norm,az_norm = norm_coord.pointing_direction

    alt_trans,az_trans = offset_to_altaz(tel_coord.x,tel_coord.y,az_tel,alt_tel)
    x,y = altaz_to_offset(az_trans,alt_trans,az_norm,alt_norm)
    x = x*u.rad
    y = y*u.rad

    representation = CartesianRepresentation(x.to(tel_coord.x.unit),y.to(tel_coord.x.unit),0*tel_coord.x.unit)

    return norm_coord.realize_frame(representation)

@frame_transform_graph.transform(FunctionTransform, NominalFrame, TelescopeFrame)
def nominal_to_telescope(norm_coord,tel_coord):

    alt_tel,az_tel = tel_coord.pointing_direction
    alt_norm,az_norm = norm_coord.pointing_direction

    alt_trans,az_trans = offset_to_altaz(norm_coord.x,norm_coord.y,az_norm,alt_norm)
    x,y = altaz_to_offset(az_trans,alt_trans,az_tel,alt_tel)
    x = x*u.rad
    y = y*u.rad

    representation = CartesianRepresentation(x.to(norm_coord.x.unit),y.to(norm_coord.x.unit),0*norm_coord.x.unit)

    return tel_coord.realize_frame(representation)


##############################################################


@frame_transform_graph.transform(FunctionTransform, CameraFrame, TelescopeFrame)
def camera_to_telescope(camera_coord, telescope_frame):

    x_pos = camera_coord.cartesian.x
    y_pos = camera_coord.cartesian.y

    rot = telescope_frame.rotation
    if rot ==0:
        x=x_pos
        y=y_pos
    else:
        x = x_pos*cos(rot) - y_pos*sin(rot)
        y = y_pos*sin(rot) + y_pos*cos(rot)

    f = telescope_frame.focal_length

    x = (x/f) * u.deg
    y = (y/f) * u.deg
    representation = CartesianRepresentation(x,y,0*u.deg)

    return telescope_frame.realize_frame(representation)


@frame_transform_graph.transform(FunctionTransform, TelescopeFrame, CameraFrame)
def telescope_to_camera(telescope_coord, camera_frame):

    x_pos = telescope_coord.cartesian.x
    y_pos = telescope_coord.cartesian.y
    rot = telescope_coord.rotation * -1

    if rot ==0:
        x=x_pos
        y=y_pos
    else:
        x = x_pos*cos(rot) - y_pos*sin(rot)
        y = y_pos*sin(rot) + y_pos*cos(rot)

    f = telescope_coord.focal_length

    x = x*(f/u.m)  # Remove distance units here as we are using small angle approx
    y = y*(f/u.m)

    representation = CartesianRepresentation(x,y,0*u.deg)

    return camera_frame.realize_frame(representation)


############### Ground and Tilted system #####################

class TiltedGroundFrame(BaseCoordinateFrame):
    """Tilted telescope coordinate frame.
    """
    default_representation = CartesianRepresentation
    pointing_direction = FrameAttribute(default=None)
    # time?


class GroundFrame(BaseCoordinateFrame):
    """Ground coordinate frame.
    """
    default_representation = CartesianRepresentation


def get_shower_trans_matrix (azimuth,altitude):
    print(altitude)
    cos_z = sin(altitude)
    sin_z = cos(altitude)
    cos_az = cos(azimuth)
    sin_az = sin(azimuth)

    trans = np.zeros([3,3])
    trans[0][0] = cos_z*cos_az
    trans[1][0] = sin_az
    trans[2][0] = sin_z*cos_az

    trans[0][1] = -cos_z*sin_az
    trans[1][1] = cos_az
    trans[2][1] = -sin_z*sin_az

    trans[0][2] = -sin_z
    trans[1][2] = 0.
    trans[2][2] = cos_z

    return trans


@frame_transform_graph.transform(FunctionTransform, GroundFrame, TiltedGroundFrame)
def ground_to_tilted(ground_coord, tilted_coord):

    x_grd = ground_coord.cartesian.x
    y_grd = ground_coord.cartesian.y
    z_grd = ground_coord.cartesian.z

    alt,az = tilted_coord.pointing_direction
    alt = alt.to(u.rad)
    az = az.to(u.rad)
    trans = get_shower_trans_matrix(az,alt)

    x_tilt = trans[0][0]*x_grd + trans[0][1]*y_grd + trans[0][2]*z_grd
    y_tilt = trans[1][0]*x_grd + trans[1][1]*y_grd + trans[1][2]*z_grd
    z_tilt = 0.0 * u.m

    representation = CartesianRepresentation(x_tilt,y_tilt,z_tilt)

    return tilted_coord.realize_frame(representation)

@frame_transform_graph.transform(FunctionTransform, TiltedGroundFrame, GroundFrame)
def tilted_to_ground(tilted_coord,ground_coord):

    x_tilt = tilted_coord.cartesian.x
    y_tilt = tilted_coord.cartesian.y

    alt,az = tilted_coord.pointing_direction
    alt = alt.to(u.rad)
    az = az.to(u.rad)

    trans = get_shower_trans_matrix(az,alt)

    x_grd = trans[0][0] * x_tilt + trans[1][0] * y_tilt
    y_grd = trans[0][1] * x_tilt + trans[1][1] * y_tilt
    z_grd = trans[0][2] * x_tilt + trans[1][2] * y_tilt

    representation = CartesianRepresentation(x_grd,y_grd,z_grd)

    return ground_coord.realize_frame(representation)
