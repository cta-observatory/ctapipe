import astropy.units as u
import numpy as np
from astropy.coordinates import spherical_to_cartesian
from astropy.utils.decorators import lazyproperty
from scipy.integrate import quad
from scipy.spatial.transform import Rotation as R
from scipy.stats import multivariate_normal

__all__ = [
    "GaussianShowermodel",
]


class GaussianShowermodel:
    @u.quantity_input(
        x=u.m,
        y=u.m,
        azimuth=u.deg,
        altitude=u.deg,
        h_max=u.m,
        width=u.m,
        length=u.m,
    )
    def __init__(self, total_photons, x, y, azimuth, altitude, h_max, width, length):
        """Create a 3D gaussian shower model for imaging.
        This is based on https://arxiv.org/pdf/astro-ph/0601373.pdf.

        Parameters
        ----------
        total_photons : int
            Number of cherenkov photons in shower
        x : u.Quantity[length]
            x coord of shower intersection on ground
        y : u.Quantity[length]
            y coord of shower intersection on ground
        azimuth : u.Quantity[angle]
            azimuthal angle relative to the shower core
        altitude : u.Quantity[angle]
            altitude relative to the shower core
        h_max : u.Quantity[length]
            height of the barycenter above ground
        width : u.Quantity[length]
            width of the shower
        length : u.Quantity[length]
            length of the shower
        """
        self.total_photons = total_photons
        self.x = x
        self.y = y
        self.azimuth = azimuth
        self.altitude = altitude
        self.zenith = 90 * u.deg - altitude
        self.h_max = h_max
        self.width = width
        self.length = length

    def density(self, x, y, z):
        """Evaluate 3D gaussian."""
        return self.total_photons * self._gauss.pdf(np.array([x, y, z]))

    @lazyproperty
    def _gauss(self):
        """Calculate 3d gaussian with barycenter as the mean and width and height in the covariance matrix."""
        # Rotate covariance matrix
        cov = np.zeros((3, 3)) * u.m
        cov[0, 0] = self.width
        cov[1, 1] = self.width
        cov[2, 2] = self.length

        r = R.from_rotvec(
            [0, self.zenith.to_value(u.rad), self.azimuth.to_value(u.rad)]
        )
        cov = r.as_matrix().T @ cov @ r.as_matrix()

        gauss = multivariate_normal(
            mean=self.barycenter.to_value(u.m), cov=cov.to_value(u.m)
        )

        return gauss

    @lazyproperty
    def barycenter(self):
        """Calculates barycenter of the shower.
        This is given by vector pointing to the impact on ground + the vector of the shower with azimuth and zenith at h_max.
        """
        b = np.zeros(3) * u.m
        b[0] = (
            self.h_max
            * np.cos(self.azimuth.to_value(u.rad))
            * np.tan(self.zenith.to_value(u.rad))
            + self.x
        )
        b[1] = (
            self.h_max
            * np.sin(self.azimuth.to_value(u.rad))
            * np.tan(self.zenith.to_value(u.rad))
            + self.y
        )
        b[2] = self.h_max
        return b

    def photon_integral(self, vec_oc, vec_los, epsilon):
        """Solves the photon integral according to https://arxiv.org/pdf/astro-ph/0601373.pdf Appendix 1 Equation (5).

        Parameters
        ----------
        vec_oc : u.Quantity[length]
            3d vector between optical center of telescope and barycenter of the shower
        vec_los : u.Quantity[length]
            Vector for each pixel along the line of sight as a 1d-quantity of shape (n_pixels)
        epsilon : u.Quantity[Angle]
            Angle between the viewing direction and shower axis for each pixel as a 1d-quantity of shape (n_pixels)
        """
        ce = np.cos(epsilon)
        sig_L = self.length
        sig_T = self.width

        sig_u_sq = sig_T**2 * ce**2 + sig_L**2 * (1 - ce**2)
        sig_D_sq = sig_L**2 - sig_T**2

        B_p = np.einsum("i,ni->n", vec_oc, vec_los)
        B_s = np.einsum("i,i->", vec_oc, self.vec_shower_axis)

        delta_B_sq = np.einsum("i,i->", vec_oc, vec_oc) - B_p**2
        upper_bound = -(
            (sig_L**2 * B_p - sig_D_sq * ce * B_s)
            / (np.sqrt(sig_u_sq) * sig_T * sig_L)
        ).to_value(u.dimensionless_unscaled)

        C = 1 - np.vectorize(self._freq)(upper_bound)
        constant = self.total_photons * C / (2 * np.pi * np.sqrt(sig_u_sq) * sig_T)

        return constant * np.exp(
            -0.5
            * (
                delta_B_sq / sig_T**2
                - sig_D_sq / (sig_T**2 * sig_u_sq) * (ce * B_p - B_s) ** 2
            )
        )

    def _freq(self, x):
        """Helper function."""
        return (
            1
            / np.sqrt(2 * np.pi)
            * quad(lambda t: np.exp(-(t**2) / 2), -np.inf, x)[0]
        )

    @lazyproperty
    def vec_shower_axis(self):
        """Calculates the unit vector of the shower axis."""
        x, y, z = spherical_to_cartesian(1, lat=self.altitude, lon=self.azimuth)
        vec = np.stack((x, y, z), -1)
        return vec

    def emission_probability(self, epsilon):
        """Calculates the emission probability of a photon with angle epsilon to the shower axis. https://arxiv.org/pdf/astro-ph/0601373.pdf Assumption 2.2.2
        Parameters
        ----------
        epsilon : u.Quantity[Angle]
            Angle between viewing direction and shower axis for each pixel as a 1d-quantity of shape (n_pixels)
        """
        eta = 15 * u.mrad * np.sqrt(np.cos(self.zenith.to_value(u.rad)))  # 15mrad

        normalization = 1 / (9 * np.pi * eta**2)

        proba = np.zeros(epsilon.shape) * normalization.unit
        proba[epsilon < eta] = normalization
        proba[epsilon >= eta] = (
            normalization
            * eta
            / epsilon[epsilon >= eta]
            * np.exp(-(epsilon[epsilon >= eta] - eta) / (4 * eta))
        )

        return proba
