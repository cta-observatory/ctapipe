from math import log

import numpy as np
from astropy import units as u
from ctapipe.utils.fitshistogram import Histogram
from scipy import ndimage
from ctapipe.instrument import get_atmosphere_profile_functions
from scipy.optimize import fsolve


class ShowerMaxEstimator:
    """
    Class that calculates the height of the shower maximum
    given a parametrisation of the atmosphere
    and certain parameters of the shower itself

    Parameters
    ----------
    atmosphere_profile_name : string
       path to text file that contains a table of the
       atmosphere parameters
    """

    def __init__(self, atmosphere_profile_name):

        self.thickness_profile, self.altitude_profile = \
            get_atmosphere_profile_functions(atmosphere_profile_name)

    def find_shower_max_height(self, energy, h_first_int, gamma_alt):
        """
        estimates the height of the shower maximum in the atmosphere
        according to equation (3) in [arXiv:0907.2610v3]

        Parameters
        ----------
        energy : astropy.Quantity
            energy of the parent gamma photon
        h_first_int : astropy.Quantity
            hight of the first interaction
        gamma_alt : astropy.Quantity or float
            altitude / pi-minus-zenith (in radians in case of float)
            of the parent gamma photon

        Returns
        -------
        shower_max_height : astropy.Quantity
            height of the shower maximum
        """

        # offset of the shower-maximum in radiation lengths
        c = 0.97 * log(energy / (83 * u.MeV)) - 1.32
        # radiation length in dry air at 1 atm = 36,62 g / cm**2 [PDG]
        c *= 36.62 * u.g * u.cm ** -2
        # showers with a more horizontal direction spend more path
        # length in each atm. layer the "effective transverse
        # thickness" they have to pass is reduced
        c *= np.sin(gamma_alt)

        # find the thickness at the height of the first interaction
        t_first_int = self.thickness_profile(h_first_int)

        # total thickness at shower maximum = thickness at first
        # interaction + thickness traversed to shower maximum
        t_shower_max = t_first_int + c

        # now find the height with the wanted thickness by solving for the
        # desired thickness
        return self.altitude_profile(t_shower_max)
