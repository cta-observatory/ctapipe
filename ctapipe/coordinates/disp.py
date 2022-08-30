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

__all__ = ["horizontal_to_telescope", "telescope_to_horizontal", "DispConverter"]


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


class DispConverter(Component):
    """Convert (norm, sign) predictions into (alt, az) predictions"""

    # This is a temporary solution for simulations using a single pointing position
    pointing_altitude = u.Quantity
    pointing_azimuth = u.Quantity

    def __call__(
        self, event: ArrayEventContainer, regressor_cls, classifier_cls
    ) -> None:
        """Convert and fill in corresponding container"""
        prefix = regressor_cls + "_" + classifier_cls

        for tel_id in event.trigger.tels_with_trigger:
            # Maybe NormContainer + SignContainer makes more sense
            # As is: DispContainer always contains only sign or norm
            norm = event.dl2.tel[tel_id].disp[regressor_cls].norm
            valid_norm = event.dl2.tel[tel_id].disp[regressor_cls].norm_is_valid
            sign = event.dl2.tel[tel_id].disp[classifier_cls].sign
            valid_sign = event.dl2.tel[tel_id].disp[classifier_cls].sign_is_valid

            if valid_sign:
                sign = -1 if sign < 0.5 else 1

            disp = norm * sign

            fov_lon = event.dl1.tel[tel_id].parameters.hillas.fov_lon + disp * np.cos(
                event.dl1.tel[tel_id].parameters.hillas.psi.to(u.rad)
            )
            fov_lat = event.dl1.tel[tel_id].parameters.hillas.fov_lat + disp * np.sin(
                event.dl1.tel[tel_id].parameters.hillas.psi.to(u.rad)
            )
            alt, az = telescope_to_horizontal(
                lon=fov_lon,
                lat=fov_lat,
                pointing_alt=event.pointing.tel[tel_id].altitude.to(u.deg),
                pointing_az=event.pointing.tel[tel_id].azimuth.to(u.deg),
            )

            event.dl2.tel[tel_id].geometry[prefix] = ReconstructedGeometryContainer(
                alt=alt, az=az, is_valid=np.logical_and(valid_sign, valid_norm)
            )

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

        fov_lon = table["hillas_fov_lon"] + disp_predictions * np.cos(
            table["hillas_psi"].to(u.rad)
        )
        fov_lat = table["hillas_fov_lat"] + disp_predictions * np.sin(
            table["hillas_psi"].to(u.rad)
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
