#!/usr/bin/env python3
"""

"""
import math
import numpy as np
from iminuit import Minuit
from astropy import units as u
from ctapipe.reco.template_interpolator import TemplateInterpolator


class ImPACTFitter(object):
    """
    """
    def __init__(self):

        self.pixel_x = 0
        self.pixel_y = 0
        self.image = 0
        self.ped = 0
        self.spe = 0

        self.prediction = dict()

        self.prediction["LST"] = \
            TemplateInterpolator("/Users/dparsons/Documents/Unix/CTA/ImPACT_pythontests/LST.obj")

        self.tel_pos_x = 0
        self.tel_pos_y = 0

        self.peak_x = 0
        self.peak_y = 0

        self.unit = u.deg

    def get_brightest_mean(self, num_pix=3):

        peak_x = list()
        peak_y = list()

        for im, px, py in zip(self.image, self.pixel_x, self.pixel_y):
            top_index = im.argsort()[-1*num_pix:][::-1]
            weight = im[top_index]
            weighted_x = px[top_index] * weight
            weighted_y = py[top_index] * weight

            peak_x.append(np.sum(weighted_x)/np.sum(weight))
            peak_y.append(np.sum(weighted_y)/np.sum(weight))

        self.peak_x = peak_x
        self.peak_y = peak_y

    def image_prediction(self, type, zenith, azimuth, energy, impact, x_max, pix_x, pix_y):
        """

        Parameters
        ----------
        type
        zenith
        azimuth
        energy
        impact
        x_max
        pix_x
        pix_y

        Returns
        -------

        """

        return self.prediction[type].interpolate([energy, impact, x_max], pix_x, pix_y)

    def get_likelihood(self, type, zenith, azimuth, energy, impact, x_max):
        """

        Parameters
        ----------
        type
        zenith
        azimuth
        energy
        impact
        x_max

        Returns
        -------

        """
        sum_like = 0
        for tel_count in range(len(type)):
            prediction = self.image_prediction(type[tel_count], zenith[tel_count], azimuth[tel_count],
                                               energy[tel_count], impact[tel_count], x_max[tel_count])

            like = self.calc_likelihood(self.image[tel_count], prediction, self.spe[tel_count], self.ped[tel_count])
            sum_like += np.sum(like)

        return -2 * sum_like

    @staticmethod
    def calc_likelihood(image, prediction, spe_width, ped):
        """
        Calculate likelihood of prediction given the measured signal, gaussian approx from
        de Naurois et al 2009

        Parameters
        ----------
        image: ndarray
            Pixel amplitudes from image
        prediction: ndarray
            Predicted pixel amplitudes from model
        spe_width: ndarray
            width of single p.e. distributio
        ped: ndarray
            width of pedestal

        Returns
        -------
        ndarray: likelihood for each pixel
        """
        sq = 1./np.sqrt(2 * math.pi * (np.power(ped ,2)
                                       + prediction * (1 + pow(spe_width, 2))))

        diff = np.power(image - prediction, 2.)
        denom = 2 * (np.power(ped ,2) + prediction * (1 + pow(spe_width, 2)))
        expo = np.exp(-1 * diff / denom)
        sm = expo<1e-300
        expo[sm] = 1e-300
        return np.log(sq*expo)

    def fit_event(self, image, pixel_x, pixel_y, tel_x, tel_y, zenith, az):
        """

        """

        # First store these parameters in the class so we can use them in minimisation
        self.image = image
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y
        self.unit = pixel_x.unit

        self.get_brightest_mean(num_pix=3)
        
        # Create Minuit object with first guesses at parameters, strip away the units as Minuit doesnt like them
        #min = Minuit(self.likelihood,impact_dist=4,limit_impact_dist=(0,25),error_impact_dist=5,
        #             phi=0,limit_phi=(-1*math.pi,1*math.pi),error_phi=0.1,
        #             radius=radius.value,fix_radius=True,centre_x=centre_x.value,fix_centre_x=True,
        #             centre_y=centre_y.value,fix_centre_y=True,
        #             width=0.1,error_width = 0.001*radius.value,limit_width=(0,1),
        #             eff=0.1,error_eff=0.05,limit_eff=(0,1),
        #             errordef=1)

        # Perform minimisation
        #min.migrad()
        # Get fitted values
        #fit_params = min.values
        #print(fit_params)
        # Return interesting stuff
        #return fit_params['impact_dist']*u.m ,fit_params['phi']*u.rad,fit_params['width']*self.unit,fit_params['eff']
