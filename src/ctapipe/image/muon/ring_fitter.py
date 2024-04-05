import numpy as np
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


FIT_METHOD_BY_NAME = {m.__name__: m for m in [kundu_chaudhuri, taubin]}

__all__ = ["MuonRingFitter"]


class MuonRingFitter(Component):
    """Different ring fit algorithms for muon rings"""

    fit_method = traits.CaselessStrEnum(
        list(FIT_METHOD_BY_NAME.keys()),
        default_value=list(FIT_METHOD_BY_NAME.keys())[0],
    ).tag(config=True)

    def __call__(self, fov_lon, fov_lat, img, mask):
        """allows any fit to be called in form of
        MuonRingFitter(fit_method = "name of the fit")
        """
        fit_function = FIT_METHOD_BY_NAME[self.fit_method]
        radius, center_fov_lon, center_fov_lat = fit_function(
            fov_lon, fov_lat, img, mask
        )

        return MuonRingContainer(
            center_fov_lon=center_fov_lon,
            center_fov_lat=center_fov_lat,
            radius=radius,
            center_phi=np.arctan2(center_fov_lat, center_fov_lon),
            center_distance=np.sqrt(center_fov_lon**2 + center_fov_lat**2),
        )
