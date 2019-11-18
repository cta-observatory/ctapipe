import numpy as np
import astropy.units as u
from ctapipe.core import Component
from ctapipe.io.containers import MuonRingParameter
from .fitting import kundu_chaudhuri_circle_fit
import traitlets as traits
from iminuit import Minuit

__all__ = ['MuonRingFitter', 'muon_ring_fit']

def muon_ring_fit(fit_method, *args, **kwargs):
    fitter = MuonRingFitter(fit_method)
    fit_result = fitter(*args, **kwargs)
    return fit_result

class MuonRingFitter(Component):
    """Different ring fit algorithms for muon rings
    """
    fit_method = traits.CaselessStrEnum(
        ['taubin', 'chaudhuri_kundu'],
        default_value='chaudhuri_kundu'
    ).tag(config=True)

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config, parent, **kwargs)

        self.fit_map = {
            'taubin': taubin_fit,
            'chaudhuri_kundu': chaudhuri_kundu_fit
        }

    def __call__(self, geom, img, mask):
        """allows any fit to be called in form of
            MuonRingFitter(fit_method = "name of the fit")
        """
        output = self.fit_map[self.fit_method](geom, img, mask)

        return output


def chaudhuri_kundu_fit(geom, weight, mask, times=None):
    """Fast and reliable analytical circle fitting method previously used
    in the H.E.S.S.  experiment for muon identification

    Implementation based on [chaudhuri93]_
    """
    x = geom.pix_x
    y = geom.pix_y
    original_unit = x.unit

    x = x.to_value(original_unit)
    y = y.to_value(original_unit)
    weight = weight * mask

    radius, center_x, center_y = kundu_chaudhuri_circle_fit(x, y, weight)

    output = MuonRingParameter()
    output.ring_center_x = center_x  * original_unit
    output.ring_center_y = center_y  * original_unit
    output.ring_radius = radius  * original_unit
    output.ring_phi = np.arctan(center_y / center_x)
    output.ring_inclination = np.sqrt(center_x ** 2. + center_y ** 2.)
    output.ring_fit_method = "ChaudhuriKundu"

    return output


def taubin_fit(geom, weight, mask, times=None):
    """
    Parameters
    ----------
    x: array in u.m
        X position of pixel surviving the cleaning

    y: array in u.m
        Y position of pixel surviving the cleaning

    Returns : MuonRingParameter
    -------
    """

    """ initialization for taubin_fit with the
    initial default values, errors, and limits
    """
    x = geom.pix_x
    y = geom.pix_y
    orinal_unit = x.unit

    x_min_in_m = x.min().to_value(orinal_unit)
    x_max_in_m = x.max().to_value(orinal_unit)
    y_min_in_m = y.min().to_value(orinal_unit)
    y_max_in_m = y.max().to_value(orinal_unit)

    x = x[mask]
    y = y[mask]

    taubin_r_initial = x_max_in_m / 2
    taubin_error = taubin_r_initial * 0.1
    xc = 0
    yc = 0

    # minimization method
    fit = Minuit(
        make_taubin_loss_function(
            x.to_value(orinal_unit),
            y.to_value(orinal_unit)
        ),
        xc=xc,
        yc=yc,
        r=taubin_r_initial,
        error_xc=taubin_error,
        error_yc=taubin_error,
        error_r=taubin_error,
        limit_xc=(x_min_in_m, x_max_in_m),
        limit_yc=(y_min_in_m, y_max_in_m),
        pedantic=False
    )
    fit.migrad()

    output = MuonRingParameter()
    output.ring_center_x = fit.values['xc'] * orinal_unit
    output.ring_center_y = fit.values['yc'] * orinal_unit
    output.ring_radius = fit.values['r'] * orinal_unit
    output.ring_fit_method = "Taubin"

    return output


def make_taubin_loss_function(x, y):

    def taubin_loss_function(xc, yc, r):
        """taubin fit formula
        reference : Barcelona_Muons_TPA_final.pdf (slide 6)
        """
        upper_term = (
            (
                (x - xc) ** 2 +
                (y - yc) ** 2
                - r ** 2
            ) ** 2
        ).sum()

        lower_term = (
            (
                (x - xc) ** 2 +
                (y - yc) ** 2
            )
        ).sum()

        return np.abs(upper_term) / np.abs(lower_term)

    return taubin_loss_function
