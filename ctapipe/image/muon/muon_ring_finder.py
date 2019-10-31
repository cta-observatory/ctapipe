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

    tel_type = traits.CaselessStrEnum(
        ['LST_LST_LSTCam', 'MST_MST_NectarCam', 'MST_MST_FlashCam', 'MST_SCT_SCTCam',
         'SST_1M_DigiCam', 'SST_GCT_CHEC', 'SST_ASTRI_ASTRICam', 'SST_ASTRI_CHEC'],
        default_value='LST_LST_LSTCam'
    ).tag(config=True)

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config, parent, **kwargs)

    def fitFormula(self, xc, yc, r):
        '''taubin fit formula
        reference : Barcelona_Muons_TPA_final.pdf (slide 6)
        '''
        upper_term = sum(
            (
                    (np.array(self.x) - xc) ** 2 +
                    (np.array(self.y) - yc) ** 2
                    - r ** 2
            ) ** 2
        )
        lower_term = sum(
            (
                    (np.array(self.x) - xc) ** 2 +
                    (np.array(self.y) - yc) ** 2
            )
        )

        return np.abs(upper_term) / np.abs(lower_term)

    def initialization(self):
        """ initialization for taubin_fit with the
        initial default values, errors, and limits
        """
        names = ['LST_LST_LSTCam', 'MST_MST_NectarCam', 'MST_MST_FlashCam', 'MST_SCT_SCTCam',
                 'SST_1M_DigiCam', 'SST_GCT_CHEC', 'SST_ASTRI_ASTRICam', 'SST_ASTRI_CHEC']
        taubin_r_initial = [0.02, 0.03, 0.03, 0.03, 0.03, 0.02, 0.04, 0.02]
        taubin_error = [0.003, 0.005, 0.005, 0.005, 0.006, 0.004, 0.007, 0.004]
        taubin_limit = [(-0.0365, 0.0365), (-0.0640, 0.0640), (-0.0625, 0.0625), (-0.0646, 0.0646),
                        (-0.0719, 0.0719), (-0.0462, 0.0462), (-0.0848, 0.0848)]
        taubin_params = {'Name': names, 'rInitial': taubin_r_initial,
                         'error': taubin_error, 'limit': taubin_limit}
        taubin_dictIndex = taubin_params['Name'].index(self.tel_type)
        radius = taubin_params['rInitial'][taubin_dictIndex]
        error = taubin_params['error'][taubin_dictIndex]
        limit = taubin_params['limit'][taubin_dictIndex]
        xc = 0
        yc = 0
        init_params = {
            'xc': xc,
            'yc': yc,
            'r': radius,
        }

        init_errs = {
            'error_xc': error,
            'error_yc': error,
            'error_r': error,
        }

        init_limits = {
            'limit_xc': limit,
            'limit_yc': limit,
        }
        return init_params, init_errs, init_limits

    def taubin_fit(self, x, y, weight=None, times=None):
        '''
        Parameters
        ----------
        px: array
            X position of pixel

        py: array
            Y position of pixel

        Returns
        -------
        xc: x coordinate of fitted ring center
        yc: y coordinate of fitted ring center
        r: radius of fitted ring
       '''
        init_params, init_errs, init_limits = self.initialization()
        print("initparams", init_params, init_errs, init_limits)
        # minimization method
        m = Minuit(self.fitFormula,
                   **init_params,
                   **init_errs,
                   **init_limits,
                   pedantic=False)
        m.migrad()
        # calculate xc, yc, r
        fitparams = m.values
        xc_fit = fitparams['xc']
        yc_fit = fitparams['yc']
        r_fit = fitparams['r']

        output = MuonRingParameter()
        output.ring_center_x = xc_fit * u.rad
        output.ring_center_y = yc_fit * u.rad
        output.ring_radius = r_fit * u.rad
        output.ring_fit_method = "Taubin"

        return output

    def chaudhuri_kundu_fit(self, x, y, weight, times=None):
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
        output.ring_center_x = centre_x  # *u.deg
        output.ring_center_y = centre_y  # *u.deg
        output.ring_radius = radius  # *u.deg
        output.ring_phi = np.arctan(centre_y / centre_x)
        output.ring_inclination = np.sqrt(centre_x ** 2. + centre_y ** 2.)
        # output.meta.ring_fit_method = "ChaudhuriKundu"
        output.ring_fit_method = "ChaudhuriKundu"

        return output

    def fit(self, x, y, img):
        """allows any fit to be called in form of
            MuonRingFitter(fit_method = "name of the fit")
        """
        self.x = x
        self.y = y
        fit_map = {'taubin': self.taubin_fit,
                   'chaudhuri_kundu': self.chaudhuri_kundu_fit
                   }
        output = fit_map[self.fit_method](x, y, img)

        return output