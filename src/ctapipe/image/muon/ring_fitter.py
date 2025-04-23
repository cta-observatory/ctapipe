import numpy as np
import pandas as pd
import traitlets as traits

from ctapipe.containers import MuonRingContainer
from ctapipe.core import Component

from .fitting import kundu_chaudhuri_circle_fit, taubin_circle_fit

# the fit methods do not expose the same interface, so we
# force the same interface onto them, here.
# we also modify their names slightly, since the names are
# exposed to the user via the string traitlet `fit_method`


def kundu_chaudhuri(fov_lon, fov_lat, weights, mask):
    """kundu_chaudhuri_circle_fit with fov_lon, fov_lat, weights, mask interface"""
    return kundu_chaudhuri_circle_fit(fov_lon[mask], fov_lat[mask], weights[mask])


def taubin(fov_lon, fov_lat, weights, mask):
    """taubin_circle_fit with fov_lon, fov_lat, weights, mask interface"""
    return taubin_circle_fit(fov_lon, fov_lat, mask)


def kundu_chaudhuri_taubin(fov_lon, fov_lat, weights, mask):
    """taubin_circle_fit with fov_lon, fov_lat, weights, mask interface
    with initial parameters provided by kundu_chaudhuri"""
    print("kundu_chaudhuri_taubin")
    taubin_r_initial, xc_initial, yc_initial = kundu_chaudhuri(
        fov_lon, fov_lat, weights, mask
    )
    return taubin_circle_fit(
        fov_lon,
        fov_lat,
        mask,
        weights,
        taubin_r_initial,
        xc_initial,
        yc_initial,
    )


FIT_METHOD_BY_NAME = {
    m.__name__: m for m in [kundu_chaudhuri, taubin, kundu_chaudhuri_taubin]
}

__all__ = ["MuonRingFitter"]


class MuonRingFitter(Component):
    """Different ring fit algorithms for muon rings"""

    FIT_COUNTER = 0

    fit_method = traits.CaselessStrEnum(
        list(FIT_METHOD_BY_NAME.keys()),
        default_value=list(FIT_METHOD_BY_NAME.keys())[2],
    ).tag(config=True)

    def __call__(self, fov_lon, fov_lat, img, mask):
        """allows any fit to be called in form of
        MuonRingFitter(fit_method = "name of the fit")
        """
        fit_function = FIT_METHOD_BY_NAME[self.fit_method]
        radius, center_fov_lon, center_fov_lat = fit_function(
            fov_lon, fov_lat, img, mask
        )

        """temporary function to be removed"""
        self.print_to_csv_lon_lat(
            fov_lon, fov_lat, img, mask, radius, center_fov_lon, center_fov_lat
        )

        return MuonRingContainer(
            center_fov_lon=center_fov_lon,
            center_fov_lat=center_fov_lat,
            radius=radius,
            center_phi=np.arctan2(center_fov_lat, center_fov_lon),
            center_distance=np.sqrt(center_fov_lon**2 + center_fov_lat**2),
        )

    """temporary function to be removed"""

    def print_to_csv_lon_lat(
        self, fov_lon, fov_lat, img, mask, radius, center_fov_lon, center_fov_lat
    ):
        fov_lon_deg = fov_lon.to_value(unit=fov_lon.unit)
        fov_lat_deg = fov_lat.to_value(unit=fov_lat.unit)
        radius_fit = np.ones(len(fov_lat_deg)) * radius.to_value(unit=radius.unit)
        center_fov_lon_fit = np.ones(len(fov_lat_deg)) * center_fov_lon.to_value(
            unit=center_fov_lon.unit
        )
        center_fov_lat_fit = np.ones(len(fov_lat_deg)) * center_fov_lat.to_value(
            unit=center_fov_lat.unit
        )
        print("FIT_COUNTER     = ", self.FIT_COUNTER)
        print("radius          = ", radius)
        print("center_fov_lon  = ", center_fov_lon)
        print("center_fov_lat  = ", center_fov_lat)
        csv_val_out = "./"
        csv_val_out += str(self.FIT_COUNTER)
        csv_val_out += ".csv"
        pd.DataFrame(
            np.column_stack(
                (
                    fov_lon_deg,
                    fov_lat_deg,
                    img,
                    mask.astype(int),
                    radius_fit,
                    center_fov_lon_fit,
                    center_fov_lat_fit,
                )
            ),
            columns=[
                "fov_lon",
                "fov_lat",
                "img",
                "mask",
                "radius_fit",
                "center_fov_lon_fit",
                "center_fov_lat_fit",
            ],
        ).to_csv(csv_val_out, sep=" ")
        MuonRingFitter.increment()

    """temporary function to be removed"""

    @staticmethod
    def increment():
        MuonRingFitter.FIT_COUNTER += 1
