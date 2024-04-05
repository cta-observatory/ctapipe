import astropy.units as u
from erfa.ufunc import s2p as spherical_to_cartesian

__all__ = [
    "altaz_to_righthanded_cartesian",
]


def altaz_to_righthanded_cartesian(alt, az):
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
    if hasattr(az, "unit"):
        az = az.to_value(u.rad)
        alt = alt.to_value(u.rad)

    return spherical_to_cartesian(-az, alt, 1.0)
