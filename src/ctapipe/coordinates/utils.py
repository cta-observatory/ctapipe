import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz
from erfa.ufunc import p2s as cartesian_to_spherical
from erfa.ufunc import s2p as spherical_to_cartesian

from .ground_frames import _get_xyz

__all__ = [
    "altaz_to_righthanded_cartesian",
    "get_point_on_shower_axis",
]


_zero_m = u.Quantity(0, u.m)


def altaz_to_righthanded_cartesian(alt, az, distance=1.0):
    """Turns an alt/az coordinate into a 3d direction in a right-handed coordinate
    system.  This is because azimuths are in a left-handed system.

    See e.g: https://github.com/astropy/astropy/issues/5300

    Parameters
    ----------
    alt: u.Quantity
        altitude
    az: u.Quantity
        azimuth
    """
    # this is an optimization, shaves of ca. 50 % compared to handing over units
    # to the erfa function
    if hasattr(az, "unit"):
        az = az.to_value(u.rad)
        alt = alt.to_value(u.rad)

    unit = None
    if hasattr(distance, "unit"):
        unit = distance.unit
        distance = distance.value

    result = spherical_to_cartesian(-az, alt, distance)
    if unit is not None:
        return u.Quantity(result, unit=unit, copy=False)
    return result


@u.quantity_input(core_x=u.m, core_y=u.m, alt=u.rad, az=u.rad, distance=u.km)
def get_point_on_shower_axis(core_x, core_y, alt, az, telescope_position, distance):
    """
    Get a point on the shower axis in AltAz frame as seen by a telescope at the given position.

    Parameters
    ----------
    core_x : u.Quantity[length]
        Impact x-coordinate
    core_y : u.Quantity[length]
        Impact y-coordinate
    alt : u.Quantity[angle]
        Altitude of primary
    az : u.Quantity[angle]
        Azimuth of primary
    telescope_position : GroundFrame
        Telescope position
    distance : u.Quantity[length]
        Distance along the shower axis from the ground of the point returned.

    Returns
    -------
    coord : AltAz
        The AltAz coordinate of a point on the shower axis at the given distance
        from the impact point.
    """
    impact = u.Quantity([core_x, core_y, _zero_m], unit=u.m)
    # move up the shower axis by slant_distance
    point = impact + altaz_to_righthanded_cartesian(alt=alt, az=az, distance=distance)

    # offset by telescope positions and convert to sperical
    # to get local AltAz for each telescope
    cartesian = point[np.newaxis, :] - _get_xyz(telescope_position).T
    lon, lat, _ = cartesian_to_spherical(cartesian)
    return AltAz(alt=lat, az=-lon, copy=False)
