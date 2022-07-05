import astropy.units as u
import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial.transform import Rotation as R


__all__ = [
    "Gaussian",
]


class Gaussian:
    @u.quantity_input(
        x=u.m,
        y=u.m,
        phi=u.deg,
        theta=u.deg,
        first_interaction=u.m,
        width=u.m,
        length=u.m,
    )
    def __init__(
        self, total_photons, x, y, phi, theta, first_interaction, width, length
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
        phi : u.Quantity[angle]
            azimuthal angle defining orientation of shower
        theta : u.Quantity[angle]
            polar angle defining orientation of shower
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
        self.phi = phi
        self.theta = theta
        self.first_interaction = first_interaction
        self.width = width
        self.length = length
        self.barycenter = self.calcBC()

        # Calculate 3d gaussian with barycenter as the mean and width and height in the covariance matrix.
        # Rotate covariance matrix
        cov = np.zeros((3, 3)) * u.m
        cov[0, 0] = self.width
        cov[1, 1] = self.width
        cov[2, 2] = self.length

        r = R.from_rotvec([0, self.theta.to_value(u.rad), self.phi.to_value(u.rad)])
        cov = r.as_matrix().T @ cov @ r.as_matrix()

        self.gauss = multivariate_normal(
            mean=self.barycenter.to_value(u.m), cov=cov.to_value(u.m)
        )

    def density(self, x, y, z):
        """Evaluate 3D gaussian."""
        return self.total_photons * self.gauss.pdf(np.array([x, y, z]))

    def calcBC(self):
        """Calculates barycenter of the shower.
        This is given by the vector defined by phi and theta in spherical coords + vector pointing to the first_interaction
        minus half length back to shower center.
        """
        b = np.zeros(3) * u.m
        b[0] = (
            self.first_interaction
            * np.cos(self.phi.to_value(u.rad))
            * np.tan(self.theta.to_value(u.rad))
            + self.x
            - self.length
            / 2
            * np.cos(self.phi.to_value(u.rad))
            * np.sin(self.theta.to_value(u.rad))
        )
        b[1] = (
            self.first_interaction
            * np.sin(self.phi.to_value(u.rad))
            * np.tan(self.theta.to_value(u.rad))
            + self.y
            - self.length
            / 2
            * np.sin(self.phi.to_value(u.rad))
            * np.sin(self.theta.to_value(u.rad))
        )
        b[2] = self.first_interaction - self.length / 2 * np.cos(
            self.theta.to_value(u.rad)
        )
        return b
