from numpy import cos, sin, arctan, arctan2, arcsin, sqrt, arccos, tan
import numpy as np
import astropy.units as u

__all__ = [
    'horizon_to_offset',
    'offset_to_horizon',
    'Cartesian2D', 'UnitSpherical'
]


class Cartesian2D:

    x, y = None, None

    def separation(self, other):
        if type(self) is type(other):
            return np.sqrt(np.power(self.x-other.x, 2) + np.power(self.y-other.y, 2))
        else:
            raise TypeError("Cannot compute separation between different frame types "
                            +type(self)+" "+type(other))


class UnitSpherical:

    theta, phi = None, None

    def separation(self, other):
        if type(self) is type(other):
            x_off, y_off = horizon_to_offset(self.phi, self.theta, other.phi, other.theta)
            return np.sqrt(x_off**2 + y_off**2)
        else:
            raise TypeError("Cannot compute separation between different frame types "
                            +type(self)+" "+type(other))


class Cartesian3D:

    x, y, z = None, None, None

    def separation(self, other):
        if type(self) is type(other):
            return np.sqrt(np.power(self.x-other.x, 2) + np.power(self.y-other.y, 2) +
                           np.power(self.z-other.z, 2))
        else:
            raise TypeError("Cannot compute separation between different frame types "
                            +type(self)+" "+type(other))

# Transformations defined below this point
def horizon_to_offset(obj_azimuth, obj_altitude, azimuth, altitude):
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
    x_off,y_off: (float,float)
        Offset of the event in the reference system (in radians)
    """

    diff_az = obj_azimuth - azimuth
    cosine_obj_alt = cos(obj_altitude)

    xp0 = -cos(diff_az) * cosine_obj_alt
    yp0 = sin(diff_az) * cosine_obj_alt
    zp0 = sin(obj_altitude)

    sin_sys_alt = sin(altitude)
    cos_sys_alt = cos(altitude)

    xp1 = sin_sys_alt * xp0 + cos_sys_alt * zp0
    yp1 = yp0
    zp1 = -cos_sys_alt * xp0 + sin_sys_alt * zp0

    disp = tan(arccos(zp1))
    alpha = arctan2(yp1, xp1)

    x_off = disp * cos(alpha)
    y_off = disp * sin(alpha)

    return x_off, y_off


def offset_to_horizon(x_off, y_off, azimuth, altitude):
    """Function to convert an angular offset with regard to a give
    reference system to an an absolute altitude and azimuth (This
    function is directly lifted from read_hess)

    Parameters
    ----------
    x_off: float
        X offset of the event in the reference system
    y_off: float
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

    x_off = x_off.to(u.rad).value
    y_off = y_off.to(u.rad).value
    azimuth = azimuth.to(u.rad).value
    altitude = altitude.to(u.rad).value

    offset = sqrt(x_off * x_off + y_off * y_off)
    pos = np.where(offset == 0)  # find offset 0 positions
    if len(pos[0]) > 0:
        offset[pos] = 1e-12  # add a very small offset to prevent math errors

    atan_off = arctan(offset)

    sin_atan_off = sin(atan_off)
    xp1 = x_off * (sin_atan_off / offset)
    yp1 = y_off * (sin_atan_off / offset)
    zp1 = cos(atan_off)

    sin_obj_alt = sin(altitude)
    cos_obj_alt = cos(altitude)

    xp0 = sin_obj_alt * xp1 - cos_obj_alt * zp1
    yp0 = yp1
    zp0 = cos_obj_alt * xp1 + sin_obj_alt * zp1

    obj_altitude = arcsin(zp0)
    obj_azimuth = arctan2(yp0, -xp0) + azimuth

    if len(pos[0]) > 0:
        obj_altitude[pos] = altitude
        obj_azimuth[pos] = azimuth

    obj_altitude = obj_altitude * u.rad
    obj_azimuth = obj_azimuth * u.rad

    return obj_altitude.to(unit), obj_azimuth.to(unit)
