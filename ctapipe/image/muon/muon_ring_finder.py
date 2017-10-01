import numpy as np
import astropy.units as u
from ctapipe.image.muon.ring_fitter import RingFitter
from ctapipe.io.containers import MuonRingParameter

__all__ = ['ChaudhuriKunduRingFitter']


class ChaudhuriKunduRingFitter(RingFitter):

    @u.quantity_input
    def fit(self, x: u.deg, y: u.deg, weight, times=None):
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
        factor = x**2 + y**2

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
            np.sum(weight * ((x - centre_x)**2 + (y - centre_y)**2)) /
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
