# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utilities to generate toymodel (fake) reconstruction inputs for testing
purposes.

Examples:

.. code-block:: python

    >>> from instrument import CameraGeometry
    >>> geom = CameraGeometry.make_rectangular(20, 20)
    >>> showermodel = Gaussian(x=0.25 * u.m, y=0.0 * u.m,
    length=0.1 * u.m, width=0.02 * u.m, psi='40d')
    >>> image, signal, noise = showermodel.generate_image(geom, intensity=1000)
    >>> print(image.shape)
    (400,)


"""
import numpy as np
from ctapipe.utils import linalg
import astropy.units as u
from astropy.coordinates import Angle
from scipy.stats import multivariate_normal, skewnorm, norm
from abc import ABCMeta, abstractmethod

__all__ = [
    'Gaussian',
    'SkewedGaussian',
    'ImageModel',
]


class ImageModel(metaclass=ABCMeta):

    @u.quantity_input(x=u.m, y=u.m)
    @abstractmethod
    def pdf(self, x, y):
        '''
        Probability density function
        '''
        pass

    def generate_image(self, camera, intensity=50, nsb_level_pe=20):
        """
        Generate a randomized DL1 shower image.
        For the signal, poisson random numbers are drawn from
        the expected signal distribution for each pixel.
        For the background, for each pixel a poisson random number
        if drawn with mean `nsb_level_pe`.

        Parameters
        ----------
        camera : `ctapipe.instrument.CameraGeometry`
            camera geometry object
        intensity : int
            Total number of photo electrons to generate
        nsb_level_pe : type
            level of NSB/pedestal in photo-electrons

        Returns
        -------
        image: array with length n_pixels containing the image
        signal: only the signal part of image
        noise: only the noise part of image
        """
        expected_signal = self.expected_signal(camera, intensity)

        signal = np.random.poisson(expected_signal)
        noise = np.random.poisson(nsb_level_pe, size=signal.shape)
        image = (signal + noise) - np.mean(noise)

        return image, signal, noise

    def expected_signal(self, camera, intensity):
        '''
        Expected signal in each pixel for the given camera
        and total intensity.

        Parameters
        ----------
        camera: `ctapipe.instrument.CameraGeometry`
            camera geometry object
        intensity: int
            Total number of expected photo electrons

        Returns
        -------
        image: array with length n_pixels containing the image
        '''
        pdf = self.pdf(camera.pix_x, camera.pix_y)
        return pdf * intensity * camera.pix_area.value


class Gaussian(ImageModel):
    @u.quantity_input(x=u.m, y=u.m, length=u.m, width=u.m)
    def __init__(self, x, y, length, width, psi):
        """Create 2D Gaussian model for a shower image in a camera.

        Parameters
        ----------
        centroid : u.Quantity[length, shape=(2, )]
            position of the centroid of the shower in camera coordinates
        width: u.Quantity[length]
            width of shower (minor axis)
        length: u.Quantity[length]
            length of shower (major axis)
        psi : convertable to `astropy.coordinates.Angle`
            rotation angle about the centroid (0=x-axis)

        Returns
        -------

        a `scipy.stats` object

        """
        self.x = x
        self.y = y
        self.width = width
        self.length = length
        self.psi = psi

    @u.quantity_input(x=u.m, y=u.m)
    def pdf(self, x, y):
        '''2d probability for photon electrons in the camera plane'''
        aligned_covariance = np.array([
            [self.length.to_value(u.m)**2, 0],
            [0, self.width.to_value(u.m)**2]
        ])
        # rotate by psi angle: C' = R C R+
        rotation = linalg.rotation_matrix_2d(self.psi)
        rotated_covariance = rotation @ aligned_covariance @ rotation.T

        return multivariate_normal(
            mean=[self.x.to_value(u.m), self.y.to_value(u.m)],
            cov=rotated_covariance,
        ).pdf(np.column_stack([x.to_value(u.m), y.to_value(u.m)]))


class SkewedGaussian(ImageModel):
    '''
    A shower image that has a skewness along the major axis
    '''
    @u.quantity_input(x=u.m, y=u.m, length=u.m, width=u.m)
    def __init__(self, x, y, length, width, psi, skewness):
        """Create 2D skewed Gaussian model for a shower image in a camera.
        Skewness is only applied along the main shower axis.
        See https://en.wikipedia.org/wiki/Skew_normal_distribution and
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skewnorm.html
        for details

        Parameters
        ----------
        centroid : u.Quantity[length, shape=(2, )]
            position of the centroid of the shower in camera coordinates
        width: u.Quantity[length]
            width of shower (minor axis)
        length: u.Quantity[length]
            length of shower (major axis)
        psi : convertable to `astropy.coordinates.Angle`
            rotation angle about the centroid (0=x-axis)

        Returns
        -------

        a `scipy.stats` object

        """
        self.x = x
        self.y = y
        self.width = width
        self.length = length
        self.psi = psi
        self.skewness = skewness

    def _moments_to_parameters(self):
        '''returns loc and scale from mean, std and skewnewss'''
        # see https://en.wikipedia.org/wiki/Skew_normal_distribution#Estimation
        skew23 = np.abs(self.skewness)**(2 / 3)
        delta = np.sign(self.skewness) * np.sqrt(
            (np.pi / 2 * skew23)
            / (skew23 + (0.5 * (4 - np.pi))**(2 / 3))
        )
        a = delta / np.sqrt(1 - delta**2)
        scale = self.length.to_value(u.m) / np.sqrt(1 - 2 * delta**2 / np.pi)
        loc = - scale * delta * np.sqrt(2 / np.pi)

        return a, loc, scale

    @u.quantity_input(x=u.m, y=u.m)
    def pdf(self, x, y):
        '''2d probability for photon electrons in the camera plane'''
        mu = u.Quantity([self.x, self.y]).to_value(u.m)

        rotation = linalg.rotation_matrix_2d(-Angle(self.psi))
        pos = np.column_stack([x.to_value(u.m), y.to_value(u.m)])
        long, trans = rotation @ (pos - mu).T

        trans_pdf = norm(loc=0, scale=self.width.to_value(u.m)).pdf(trans)

        a, loc, scale = self._moments_to_parameters()

        return trans_pdf * skewnorm(a=a, loc=loc, scale=scale).pdf(long)


class RingGaussian(ImageModel):
    '''
    A shower image consisting of a ring with gaussian radial profile.
    Simplified model for a muon ring.
    '''
    @u.quantity_input(x=u.m, y=u.m, radius=u.m, sigma=u.m)
    def __init__(self, x, y, radius, sigma):
        self.x = x
        self.y = y
        self.sigma = sigma
        self.radius = radius

    @u.quantity_input(x=u.m, y=u.m)
    def pdf(self, x, y):
        '''2d probability for photon electrons in the camera plane'''

        r = np.sqrt((x - self.x)**2 + (y - self.y)**2)

        return norm(
            self.radius.to_value(u.m),
            self.sigma.to_value(u.m),
        ).pdf(r)
