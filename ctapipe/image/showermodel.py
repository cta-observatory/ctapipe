import numpy as np
from astropy.utils.decorators import lazyproperty
from erfa.ufunc import s2p as spherical_to_cartesian

from ctapipe.utils.stats import survival_function

__all__ = [
    "GaussianShowermodel",
]


class GaussianShowermodel:
    def __init__(self, total_photons, x, y, azimuth, altitude, h_max, width, length):
        """Create a 3D gaussian shower model for imaging.
        This is based on https://arxiv.org/pdf/astro-ph/0601373.pdf.

        Parameters
        ----------
        total_photons : float
            Number of cherenkov photons in shower
        x : float
            x coord of shower intersection on ground in meter
        y : float
            y coord of shower intersection on ground in meter
        azimuth : float
            azimuthal angle relative to the shower core in rad
        altitude : float
            altitude relative to the shower core in rad
        h_max : float
            height of the barycenter above ground in meter
        width : float
            width of the shower in meter
        length : float
            length of the shower in meter
        """
        self.total_photons = total_photons
        self.x = x
        self.y = y
        self.azimuth = np.deg2rad(azimuth)
        self.altitude = np.deg2rad(altitude)
        self.zenith = np.pi / 2 - self.altitude
        self.h_max = h_max
        self.width = width
        self.length = length

    @lazyproperty
    def barycenter(self):
        """Calculates barycenter of the shower.
        This is given by vector pointing to the impact on ground + the vector of the shower with azimuth and zenith at h_max.
        """
        b = np.zeros(3)
        b[0] = self.h_max * np.cos(self.azimuth) * np.tan(self.zenith) + self.x
        b[1] = self.h_max * np.sin(self.azimuth) * np.tan(self.zenith) + self.y
        b[2] = self.h_max
        return b

    def photon_integral(self, vec_oc, vec_los, epsilon):
        """Solves the photon integral according to https://arxiv.org/pdf/astro-ph/0601373.pdf Appendix 1 Equation (5).

        Parameters
        ----------
        vec_oc : ndarray
            3d vector between optical center of telescope and barycenter of the shower
        vec_los : ndarray
            Vector for each pixel along the line of sight as a 1d-array of shape (n_pixels)
        epsilon : ndarray
            Angle between the viewing direction and shower axis for each pixel as a 1d-array of shape (n_pixels)
        """
        ce = np.cos(epsilon)
        sig_L = self.length
        sig_T = self.width

        sig_u_sq = sig_T**2 * ce**2 + sig_L**2 * (1 - ce**2)
        sig_D_sq = sig_L**2 - sig_T**2

        B_p = vec_los @ vec_oc
        B_s = np.dot(vec_oc, self.vec_shower_axis)

        delta_B_sq = np.dot(vec_oc, vec_oc) - B_p**2
        upper_bound = -(
            (sig_L**2 * B_p - sig_D_sq * ce * B_s)
            / (np.sqrt(sig_u_sq) * sig_T * sig_L)
        )

        C = survival_function(upper_bound)
        constant = self.total_photons * C / (2 * np.pi * np.sqrt(sig_u_sq) * sig_T)

        return constant * np.exp(
            -0.5
            * (
                delta_B_sq / sig_T**2
                - sig_D_sq / (sig_T**2 * sig_u_sq) * (ce * B_p - B_s) ** 2
            )
        )

    @lazyproperty
    def vec_shower_axis(self):
        """Calculates the unit vector of the shower axis."""
        vec = spherical_to_cartesian(self.azimuth, self.altitude, 1.0)
        return vec

    def emission_probability(self, epsilon):
        """Calculates the emission probability of a photon with angle epsilon to the shower axis. https://arxiv.org/pdf/astro-ph/0601373.pdf Assumption 2.2.2
        Parameters
        ----------
        epsilon : ndarray
            Angle between viewing direction and shower axis for each pixel as a 1d-array of shape (n_pixels)
        """
        eta = 15 * 1e-3 * np.sqrt(np.cos(self.zenith))  # 1e-3 = mrad

        normalization = 1 / (9 * np.pi * eta**2)

        proba = np.full(epsilon.shape, normalization)
        mask = epsilon >= eta
        epsilon_masked = epsilon[mask]
        proba[mask] *= (
            eta / epsilon_masked * np.exp(-(epsilon_masked - eta) / (4 * eta))
        )

        return proba
