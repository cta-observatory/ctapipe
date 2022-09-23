"""
Helper functions and components for handling coordinate transformations
during origin reconstruction using the disp method.
"""
import warnings

import astropy.units as u
from astropy.coordinates import AltAz, SkyCoord

from ctapipe.coordinates import MissingFrameAttributeWarning, TelescopeFrame

__all__ = ["horizontal_to_telescope", "telescope_to_horizontal"]


def horizontal_to_telescope(
    alt: u.Quantity, az: u.Quantity, pointing_alt: u.Quantity, pointing_az: u.Quantity
):
    """Transform coordinates form horizontal coordinates into TelescopeFrame"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingFrameAttributeWarning)

        horizontal_coord = SkyCoord(alt=alt, az=az, frame=AltAz())
        pointing = SkyCoord(alt=pointing_alt, az=pointing_az, frame=AltAz())
        tel_frame = TelescopeFrame(telescope_pointing=pointing)

        tel_coord = horizontal_coord.transform_to(tel_frame)

    return tel_coord.fov_lon.to(u.deg), tel_coord.fov_lat.to(u.deg)


def telescope_to_horizontal(
    lon: u.Quantity, lat: u.Quantity, pointing_alt: u.Quantity, pointing_az: u.Quantity
):
    """Transform coordinates from TelescopeFrame into horizontal coordinates"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingFrameAttributeWarning)

        pointing = SkyCoord(alt=pointing_alt, az=pointing_az, frame=AltAz())
        tel_coord = TelescopeFrame(
            fov_lon=lon, fov_lat=lat, telescope_pointing=pointing
        )
        horizontal_coord = tel_coord.transform_to(AltAz())

    return horizontal_coord.alt.to(u.deg), horizontal_coord.az.to(u.deg)
