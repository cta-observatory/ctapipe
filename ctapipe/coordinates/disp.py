"""
Helper functions and components for handling coordinate transformations
during origin reconstruction using the disp method.
"""
import warnings

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, SkyCoord
from astropy.table import Table

from ctapipe.coordinates import MissingFrameAttributeWarning, TelescopeFrame

from ..containers import ArrayEventContainer
from ..core import Component

__all__ = ["horizontal_to_telescope", "telescope_to_horizontal"]


def horizontal_to_telescope(
    alt: u.Quantity, az: u.Quantity, pointing_alt: u.Quantity, pointing_az: u.Quantity
):
    """Transform coordinates form horizontal coordinates into TelescopeFrame"""
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
    """Transform coordinates from TelescopeFrame into horizontal coordinates"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingFrameAttributeWarning)

        altaz = AltAz()
        pointing = SkyCoord(alt=pointing_alt, az=pointing_az, frame=altaz)
        tel_coord = TelescopeFrame(
            fov_lon=lon, fov_lat=lat, telescope_pointing=pointing
        )
        horizontal_coord = tel_coord.transform_to(altaz)

    return horizontal_coord.alt.to(u.deg), horizontal_coord.az.to(u.deg)


class DispConverter(Component):
    """Convert (norm, sign) predictions into (alt, az) predictions"""

    # This is a temporary solution for simulations using a single pointing position
    pointing_altitude = u.Quantity
    pointing_azimuth = u.Quantity

    def __call__(self, event: ArrayEventContainer) -> None:
        """Convert and fill in corresponding container"""
        # TODO
        pass

    def predict(self, table: Table, regressor_cls, classifier_cls) -> Table:
        """Convert for a table of events"""
        prefix = regressor_cls + "_" + classifier_cls

        # convert sign score [0, 1] into actual sign {-1, 1}
        valid_sign = table[f"{classifier_cls}_sign_is_valid"]

        table[f"{classifier_cls}_sign"][valid_sign] = np.where(
            table[f"{classifier_cls}_sign"][valid_sign] < 0.5, -1, 1
        )

        disp_predictions = (
            table[f"{regressor_cls}_norm"] * table[f"{classifier_cls}_sign"]
        )

        fov_lon = (
            table["hillas_fov_lon"]
            + disp_predictions * np.cos(table["hillas_psi"].to(u.rad)) * u.deg
        )
        fov_lat = (
            table["hillas_fov_lat"]
            + disp_predictions * np.sin(table["hillas_psi"].to(u.rad)) * u.deg
        )

        alt, az = telescope_to_horizontal(
            lon=fov_lon,
            lat=fov_lat,
            pointing_alt=self.pointing_altitude,
            pointing_az=self.pointing_azimuth,
        )

        result = Table(
            {
                f"{prefix}_alt": alt,
                f"{prefix}_az": az,
                f"{prefix}_is_valid": np.logical_and(
                    table[f"{regressor_cls}_norm_is_valid"],
                    table[f"{classifier_cls}_sign_is_valid"],
                ),
            }
        )

        return result
