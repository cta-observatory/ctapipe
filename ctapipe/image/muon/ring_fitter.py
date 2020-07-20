import numpy as np
from ctapipe.core import Component
from ctapipe.containers import MuonRingContainer
from .fitting import kundu_chaudhuri_circle_fit, taubin_circle_fit
import traitlets as traits


# the fit methods do not expose the same interface, so we
# force the same interface onto them, here.
# we also modify their names slightly, since the names are
# exposed to the user via the string traitlet `fit_method`


def kundu_chaudhuri(x, y, weights, mask):
    """kundu_chaudhuri_circle_fit with x, y, weights, mask interface"""
    return kundu_chaudhuri_circle_fit(x[mask], y[mask], weights[mask])


def taubin(x, y, weights, mask):
    """taubin_circle_fit with x, y, weights, mask interface"""
    return taubin_circle_fit(x, y, mask)


FIT_METHOD_BY_NAME = {m.__name__: m for m in [kundu_chaudhuri, taubin]}

__all__ = ["MuonRingFitter"]


class MuonRingFitter(Component):
    """Different ring fit algorithms for muon rings"""

    fit_method = traits.CaselessStrEnum(
        list(FIT_METHOD_BY_NAME.keys()),
        default_value=list(FIT_METHOD_BY_NAME.keys())[0],
    ).tag(config=True)

    def __call__(self, x, y, img, mask):
        """allows any fit to be called in form of
            MuonRingFitter(fit_method = "name of the fit")
        """
        fit_function = FIT_METHOD_BY_NAME[self.fit_method]
        radius, center_x, center_y = fit_function(x, y, img, mask)

        return MuonRingContainer(
            center_x=center_x,
            center_y=center_y,
            radius=radius,
            center_phi=np.arctan2(center_y, center_x),
            center_distance=np.sqrt(center_x ** 2 + center_y ** 2),
        )
