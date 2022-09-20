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

from ..containers import ArrayEventContainer, ReconstructedGeometryContainer
from ..core import Component

__all__ = [
    "horizontal_to_telescope",
    "telescope_to_horizontal",
    "MonoDispReconstructor",
]


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


class MonoDispReconstructor(Component):
    """Convert (norm, sign) predictions into (alt, az) predictions"""

    def __call__(self, event: ArrayEventContainer) -> None:
        """Convert and fill in corresponding container"""
        prefix = "disp"

        for tel_id in event.trigger.tels_with_trigger:
            norm = event.dl2.tel[tel_id].disp[prefix].norm
            sign = event.dl2.tel[tel_id].disp[prefix].sign
            valid = event.dl2.tel[tel_id].disp[prefix].is_valid

            if valid:
                disp = norm * sign

                fov_lon = event.dl1.tel[
                    tel_id
                ].parameters.hillas.fov_lon + disp * np.cos(
                    event.dl1.tel[tel_id].parameters.hillas.psi.to(u.rad)
                )
                fov_lat = event.dl1.tel[
                    tel_id
                ].parameters.hillas.fov_lat + disp * np.sin(
                    event.dl1.tel[tel_id].parameters.hillas.psi.to(u.rad)
                )
                alt, az = telescope_to_horizontal(
                    lon=fov_lon,
                    lat=fov_lat,
                    pointing_alt=event.pointing.tel[tel_id].altitude.to(u.deg),
                    pointing_az=event.pointing.tel[tel_id].azimuth.to(u.deg),
                )

                container = ReconstructedGeometryContainer(
                    alt=alt, az=az, is_valid=True
                )
            else:
                container = ReconstructedGeometryContainer(
                    alt=u.Quantity(np.nan, u.deg, copy=False),
                    az=u.Quantity(np.nan, u.deg, copy=False),
                    is_valid=False,
                )

            event.dl2.tel[tel_id].geometry[prefix] = container

    def predict(self, table: Table, pointing_altitude, pointing_azimuth) -> Table:
        """Convert for a table of events"""
        # Pointing information is a temporary solution for simulations using a single pointing position
        prefix = "disp"

        disp_predictions = table[f"{prefix}_norm"] * table[f"{prefix}_sign"]

        fov_lon = table["hillas_fov_lon"] + disp_predictions * np.cos(
            table["hillas_psi"].to(u.rad)
        )
        fov_lat = table["hillas_fov_lat"] + disp_predictions * np.sin(
            table["hillas_psi"].to(u.rad)
        )

        alt, az = telescope_to_horizontal(
            lon=fov_lon,
            lat=fov_lat,
            pointing_alt=pointing_altitude,
            pointing_az=pointing_azimuth,
        )

        result = Table(
            {
                f"{prefix}_alt": alt,
                f"{prefix}_az": az,
            }
        )

        return result
