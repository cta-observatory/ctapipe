import astropy.units as u
import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial.transform import Rotation as R
from astropy.utils.decorators import lazyproperty


__all__ = [
    "Gaussian",
]


class Gaussian:
    @u.quantity_input(
        x=u.m,
        y=u.m,
        azimuth=u.deg,
        altitude=u.deg,
        first_interaction=u.m,
        width=u.m,
        length=u.m,
    )
    def __init__(
        self, total_photons, x, y, azimuth, altitude, first_interaction, width, length
    ):
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
        first_interaction : u.Quantity[length]
            height of the first_interaction of the gamma above ground
        width : u.Quantity[length]
            width of the shower
        length : u.Quantity[length]
            length of the shower
        """
        self.total_photons = total_photons
        self.x = x
        self.y = y
        self.azimuth = azimuth
        self.zenith = 90 * u.deg - altitude
        self.first_interaction = first_interaction
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
            mean=self._barycenter.to_value(u.m), cov=cov.to_value(u.m)
        )

        return gauss

    @lazyproperty
    def _barycenter(self):
        """Calculates barycenter of the shower.
        This is given by the vector defined by azimuth and zenith in spherical coords + vector pointing to the first_interaction
        minus half length back to shower center.
        """
        b = np.zeros(3) * u.m
        b[0] = (
            self.first_interaction
            * np.cos(self.azimuth.to_value(u.rad))
            * np.tan(self.zenith.to_value(u.rad))
            + self.x
            - self.length
            / 2
            * np.cos(self.azimuth.to_value(u.rad))
            * np.sin(self.zenith.to_value(u.rad))
        )
        b[1] = (
            self.first_interaction
            * np.sin(self.azimuth.to_value(u.rad))
            * np.tan(self.zenith.to_value(u.rad))
            + self.y
            - self.length
            / 2
            * np.sin(self.azimuth.to_value(u.rad))
            * np.sin(self.zenith.to_value(u.rad))
        )
        b[2] = self.first_interaction - self.length / 2 * np.cos(
            self.zenith.to_value(u.rad)
        )
        return b
