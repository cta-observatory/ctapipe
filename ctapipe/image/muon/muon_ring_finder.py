import numpy as np
import astropy.units as u
from ctapipe.core import Component
from ctapipe.io.containers import MuonRingParameter
import traitlets as traits
from iminuit import Minuit

__all__ = ['MuonRingFitter']

class MuonRingFitter(Component):
    """Different ring fit algorithms for muon rings
    """
    fit_method = traits.CaselessStrEnum(
        ['taubin', 'chaudhuri_kundu'],
        default_value='chaudhuri_kundu'
    ).tag(config=True)

    def __init__(self, geom=None, config=None, parent=None, **kwargs):
        """
        geom: CameraGeometry
            in form of event.inst.subarray.tel[telid].camera
        """
        super().__init__(config, parent, **kwargs)

        self.fit_map = {
            'taubin': self.taubin_fit,
            'chaudhuri_kundu': chaudhuri_kundu_fit
        }

        self.geom = geom

    def taubin_fit(self, x, y, weight=None, times=None):
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
        x_min_in_m = self.geom.pix_x.min().to_value(u.m)
        x_max_in_m = self.geom.pix_x.max().to_value(u.m)
        y_min_in_m = self.geom.pix_y.min().to_value(u.m)
        y_max_in_m = self.geom.pix_y.max().to_value(u.m)

        taubin_r_initial = x_max_in_m / 2
        taubin_error = taubin_r_initial * 0.1
        xc = 0
        yc = 0

        # minimization method
        fit = Minuit(
            make_taubin_loss_function(x, y),
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
        output.ring_center_x = fit.values['xc'] * u.m
        output.ring_center_y = fit.values['yc'] * u.m
        output.ring_radius = fit.values['r'] * u.m
        output.ring_fit_method = "Taubin"

        return output



    def fit(self, x, y, img):
        """allows any fit to be called in form of
            MuonRingFitter(fit_method = "name of the fit")
        """
        output = self.fit_map[self.fit_method](x, y, img)

        return output



def chaudhuri_kundu_fit(x, y, weight, times=None):
    """Fast and reliable analytical circle fitting method previously used
    in the H.E.S.S.  experiment for muon identification

    Implementation based on [chaudhuri93]_

    Parameters
    ----------
    x: ndarray
        X position of pixel
    y: ndarray
        Y position of pixel
    weight: ndarray
        weighting of pixel in fit

    Returns
    -------
    X position, Y position, radius, orientation and inclination of circle
    """
    # First calculate the weighted average positions of the pixels
    x = x.to_value(u.m)
    y = y.to_value(u.m)

    sum_weight = np.sum(weight)
    av_weighted_pos_x = np.sum(x * weight) / sum_weight
    av_weighted_pos_y = np.sum(y * weight) / sum_weight

    # The following notation is a bit ugly but directly references the paper notation
    factor = x ** 2 + y ** 2

    a = np.sum(weight * (x - av_weighted_pos_x) * x)
    a_prime = np.sum(weight * (y - av_weighted_pos_y) * x)

    b = np.sum(weight * (x - av_weighted_pos_x) * y)
    b_prime = np.sum(weight * (y - av_weighted_pos_y) * y)

    c = np.sum(weight * (x - av_weighted_pos_x) * factor) * 0.5
    c_prime = np.sum(weight * (y - av_weighted_pos_y) * factor) * 0.5

    nom_0 = ((a * b_prime) - (a_prime * b))
    nom_1 = ((a_prime * b) - (a * b_prime))

    # Calculate circle centre and radius
    centre_x = ((b_prime * c) - (b * c_prime)) / nom_0
    centre_y = ((a_prime * c) - (a * c_prime)) / nom_1

    radius = np.sqrt(
        # np.sum(weight * ((x - centre_x*u.deg)**2 +
        # (y - centre_y*u.deg)**2)) / # centre * u.deg ???
        np.sum(weight * ((x - centre_x) ** 2 + (y - centre_y) ** 2)) /
        sum_weight
    )

    output = MuonRingParameter()
    output.ring_center_x = centre_x  * u.m
    output.ring_center_y = centre_y  * u.m
    output.ring_radius = radius  * u.m
    output.ring_phi = np.arctan(centre_y / centre_x)
    output.ring_inclination = np.sqrt(centre_x ** 2. + centre_y ** 2.)
    output.ring_fit_method = "ChaudhuriKundu"

    return output


def make_taubin_loss_function(x, y):
    x_in_m = x.to_value(u.m)
    y_in_m = y.to_value(u.m)

    def taubin_loss_function(self, xc, yc, r):
        """taubin fit formula
        reference : Barcelona_Muons_TPA_final.pdf (slide 6)
        """
        upper_term = (
            (
                (x_in_m - xc) ** 2 +
                (y_in_m - yc) ** 2
                - r ** 2
            ) ** 2
        ).sum()

        lower_term = (
            (
                (x_in_m - xc) ** 2 +
                (y_in_m - yc) ** 2
            )
        ).sum()

        return np.abs(upper_term) / np.abs(lower_term)

    return taubin_loss_function
