"""
Helper functions for coordinate transformation
"""
import warnings

import astropy.units as u
from astropy.coordinates import AltAz, SkyCoord

from ctapipe.coordinates import MissingFrameAttributeWarning, TelescopeFrame

__all__ = ["horizontal_to_telescope", "telescope_to_horizontal"]


def horizontal_to_telescope(
    alt: u.Quantity, az: u.Quantity, pointing_alt: u.Quantity, pointing_az: u.Quantity
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingFrameAttributeWarning)

        altaz = AltAz()
        horizontal_coord = SkyCoord(alt=alt, az=az, frame=altaz)
        pointing = SkyCoord(alt=pointing_alt, az=pointing_az, frame=altaz)
        tel_frame = TelescopeFrame(telescope_pointing=pointing)

        tel_coord = horizontal_coord.transform_to(tel_frame)

    return tel_coord.fov_lon.to(u.deg), tel_coord.fov_lat.to(u.deg)


def telescope_to_horizontal(
    lon: u.Quantity, lat: u.Quantity, pointing_alt: u.Quantity, pointing_az: u.Quantity
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingFrameAttributeWarning)

        altaz = AltAz()
        pointing = SkyCoord(alt=pointing_alt, az=pointing_az, frame=altaz)
        tel_coord = TelescopeFrame(
            fov_lon=lon, fov_lat=lat, telescope_pointing=pointing
        )
        horizontal_coord = tel_coord.transform_to(altaz)

    return horizontal_coord.alt.to(u.deg), horizontal_coord.az.to(u.deg)
