"""utils to deal with coordinate transformations"""

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle

from .quantities import all_to_value

__all__ = [
    "cartesian_to_polar",
    "polar_to_cartesian",
]


@u.quantity_input(x=u.m, y=u.m)
def cartesian_to_polar(x, y, wrap_angle=True):
    """
    Convert cartesian coordinates to polar.

    Parameters
    ----------
    x : astropy.units.Quantity
        X coordinate [m].
    y : astropy.units.Quantity
        Y coordinate [m].
    wrap_angle : bool
        If True (default), the polar angle is wrapped to the range
        [0, 2pi). If False, the angle is returned in the range (-pi, pi].

    Returns
    -------
    (rho, phi) : Tuple[astropy.units.Quantity, astropy.units.Quantity]
        Radial and angular coordinates [m, radians].
    """
    x_val, y_val = all_to_value(x, y, unit=u.m)
    rho = np.sqrt(x_val**2 + y_val**2) * u.m
    phi = Angle(np.arctan2(y_val, x_val), u.rad)
    if wrap_angle:
        phi = phi.wrap_at(2 * np.pi * u.rad)
    return (rho, phi)


@u.quantity_input(rho=u.m, phi=u.rad)
def polar_to_cartesian(rho, phi):
    """
    Convert polar coordinates to cartesian.

    Parameters
    ----------
    rho : astropy.units.Quantity
        Radial coordinate [m].
    phi : astropy.units.Quantity
        Angular coordinate [radians].

    Returns
    -------
    (x, y) : Tuple[astropy.units.Quantity, astropy.units.Quantity]
        X and Y coordinates [m].
    """
    rho_val, phi_val = rho.to_value(u.m), phi.to_value(u.rad)
    x = rho_val * np.cos(phi_val) * u.m
    y = rho_val * np.sin(phi_val) * u.m
    return (x, y)
