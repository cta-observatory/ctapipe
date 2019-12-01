import inspect
import numpy as np
import astropy.units as u
from ctapipe.core import Component
from ctapipe.io.containers import MuonRingParameter
from .fitting import kundu_chaudhuri_circle_fit, taubin_circle_fit
import traitlets as traits

FIT_METHODS = [kundu_chaudhuri_circle_fit, taubin_circle_fit]
SUFFIX = '_circle_fit'
FIT_METHOD_BY_NAME = {m.__name__: m for m in FIT_METHODS}
FIT_METHOD_NAMES = [
    m.__name__[0:-len(SUFFIX)]
    for m in FIT_METHODS
]

__all__ = ['MuonRingFitter']

class MuonRingFitter(Component):
    """Different ring fit algorithms for muon rings
    """
    fit_method = traits.CaselessStrEnum(
        FIT_METHOD_NAMES,
        default_value=FIT_METHOD_NAMES[0]
    ).tag(config=True)

    def __call__(self, x, y, img, mask):
        """allows any fit to be called in form of
            MuonRingFitter(fit_method = "name of the fit")
        """
        fit_function = FIT_METHOD_BY_NAME[self.fit_method + SUFFIX]
        radius, center_x, center_y = fit_function(
            x=x,
            y=y,
            weights=img,
            mask=mask
        )

        output = MuonRingParameter()
        output.ring_center_x = center_x
        output.ring_center_y = center_y
        output.ring_radius = radius
        output.ring_phi = np.arctan2(center_y, center_x)
        output.ring_inclination = np.sqrt(center_x ** 2. + center_y ** 2.)
        output.ring_fit_method = self.fit_method

        return output
