import numpy as np
import astropy.units as u
from ctapipe.core import Component
from ctapipe.io.containers import MuonRingParameter
from .fitting import kundu_chaudhuri_circle_fit, taubin_circle_fit
import traitlets as traits

__all__ = ['MuonRingFitter']

class MuonRingFitter(Component):
    """Different ring fit algorithms for muon rings
    """
    fit_method = traits.CaselessStrEnum(
        ['taubin', 'chaudhuri_kundu'],
        default_value='chaudhuri_kundu'
    ).tag(config=True)

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config, parent, **kwargs)


    def __call__(self, geom, img, mask):
        """allows any fit to be called in form of
            MuonRingFitter(fit_method = "name of the fit")
        """
        x = geom.pix_x
        y = geom.pix_y

        if self.fit_method == 'taubin':
            radius, center_x, center_y = taubin_circle_fit(x, y, mask)
        else:
            radius, center_x, center_y = kundu_chaudhuri_circle_fit(x, y, img * mask)

        output = fill_output_container(radius, center_x, center_y)
        output.ring_fit_method = self.fit_method

        return output


def fill_output_container(radius, center_x, center_y):
    output = MuonRingParameter()
    output.ring_center_x = center_x
    output.ring_center_y = center_y
    output.ring_radius = radius
    output.ring_phi = np.arctan2(center_y, center_x)
    output.ring_inclination = np.sqrt(center_x ** 2. + center_y ** 2.)

    return output
