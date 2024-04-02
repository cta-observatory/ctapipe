# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utilities to generate toymodel (fake) reconstruction inputs for testing
purposes.

Examples:

.. code-block:: python

    >>> from ctapipe.instrument import CameraGeometry
    >>> geom = CameraGeometry.make_rectangular(20, 20)
    >>> showermodel = Gaussian(
    ...    x=0.25 * u.m, y=0.0 * u.m,
    ...    length=0.1 * u.m, width=0.02 * u.m,
    ...    psi='40d'
    ... )
    >>> image, signal, noise = showermodel.generate_image(geom, intensity=1000)
    >>> print(image.shape)
    (400,)
"""

from abc import ABCMeta, abstractmethod

import astropy.units as u
import numpy as np
from numpy.random import default_rng
from scipy.ndimage import convolve1d
from scipy.stats import multivariate_normal, norm, skewnorm

from ctapipe.calib.camera.gainselection import GainChannel
from ctapipe.image.hillas import camera_to_shower_coordinates
from ctapipe.utils import linalg

__all__ = [
    "WaveformModel",
    "Gaussian",
    "SkewedGaussian",
    "ImageModel",
    "obtain_time_image",
]


TOYMODEL_RNG = default_rng(0)
VALID_GAIN_CHANNEL = {"HIGH", "LOW", "ALL"}


def obtain_time_image(x, y, centroid_x, centroid_y, psi, time_gradient, time_intercept):
    """Create a pulse time image for a toymodel shower. Assumes the time development
    occurs only along the longitudinal (major) axis of the shower, and scales
    linearly with distance along the axis.

    Parameters
    ----------
    x : u.Quantity[length]
        X camera coordinate to evaluate the time at.
        Usually the array of pixel X positions
    y : u.Quantity[length]
        Y camera coordinate to evaluate the time at.
        Usually the array of pixel Y positions
    centroid_x : u.Quantity[length]
        X camera coordinate for the centroid of the shower
    centroid_y : u.Quantity[length]
        Y camera coordinate for the centroid of the shower
    psi : convertible to `astropy.coordinates.Angle`
        rotation angle about the centroid (0=x-axis)
    time_gradient : u.Quantity[time/angle]
        Rate at which the time changes with distance along the shower axis
    time_intercept : u.Quantity[time]
        Pulse time at the shower centroid

    Returns
    -------
    float or ndarray
        Pulse time in nanoseconds at (x, y)

    """
    unit = x.unit
    x = x.to_value(unit)
    y = y.to_value(unit)
    centroid_x = centroid_x.to_value(unit)
    centroid_y = centroid_y.to_value(unit)
    psi = psi.to_value(u.rad)
    time_gradient = time_gradient.to_value(u.ns / unit)
    time_intercept = time_intercept.to_value(u.ns)

    longitudinal, _ = camera_to_shower_coordinates(x, y, centroid_x, centroid_y, psi)
    return longitudinal * time_gradient + time_intercept


class WaveformModel:
    @u.quantity_input(reference_pulse_sample_width=u.ns, sample_width=u.ns)
    def __init__(self, reference_pulse, reference_pulse_sample_width, sample_width):
        """Generate a toy model waveform for each gain channel using the reference
        pulse shape of a camera. Useful for testing image extraction algorithms.

        Does not include the electronic noise and the Excess Noise Factor of
        the photosensor, and therefore should not be used to make charge
        resolution conclusions about a camera.

        Parameters
        ----------
        reference_pulse_sample_width : u.Quantity[time]
            Sample width of the reference pulse shape
        reference_pulse : ndarray
            Reference pulse shape for each channel
        sample_width : u.Quantity[time]
            Sample width of the waveform

        """
        self.n_channels = reference_pulse.shape[-2]
        self.upsampling = 10
        reference_pulse_sample_width = reference_pulse_sample_width.to_value(u.ns)
        sample_width_ns = sample_width.to_value(u.ns)
        ref_max_sample = reference_pulse[0].size * reference_pulse_sample_width
        reference_pulse_x = np.arange(0, ref_max_sample, reference_pulse_sample_width)
        self.ref_width_ns = sample_width_ns / self.upsampling
        self.ref_interp_x = np.arange(0, reference_pulse_x.max(), self.ref_width_ns)
        self.ref_interp_y = np.zeros((self.n_channels, self.ref_interp_x.size))
        for channel in range(self.n_channels):
            self.ref_interp_y[channel] = np.interp(
                self.ref_interp_x, reference_pulse_x, reference_pulse[channel]
            )
        self.ref_interp_y = (
            self.ref_interp_y.T / (self.ref_interp_y.sum(-1) * self.ref_width_ns)
        ).T
        self.origin = self.ref_interp_y.argmax(-1) - self.ref_interp_y[0].size // 2

    def get_waveform(self, charge, time, n_samples):
        """Obtain the waveform toy model.

        Parameters
        ----------
        charge : ndarray
            Amount of charge in each pixel
            Shape: (n_pixels)
        time : ndarray
            The signal time in the waveform in nanoseconds
            Shape: (n_pixels)
        n_samples : int
            Number of samples in the waveform

        Returns
        -------
        waveform : ndarray
            Toy model waveform
            Shape (n_channels, n_pixels, n_samples)

        """
        n_pixels = charge.size
        n_upsampled_samples = n_samples * self.upsampling
        readout = np.zeros((n_pixels, n_upsampled_samples))

        sample = (time / self.ref_width_ns).astype(np.int64)
        outofrange = (sample < 0) | (sample >= n_upsampled_samples)
        sample[outofrange] = 0
        charge[outofrange] = 0
        readout[np.arange(n_pixels), sample] = charge
        convolved = np.zeros((self.n_channels, n_pixels, n_upsampled_samples))
        for channel in range(self.n_channels):
            convolved[channel] = convolve1d(
                readout,
                self.ref_interp_y[channel],
                mode="constant",
                origin=self.origin[channel],
            )
        sampled = (
            convolved.reshape(
                (
                    self.n_channels,
                    n_pixels,
                    convolved.shape[-1] // self.upsampling,
                    self.upsampling,
                )
            ).sum(-1)
            * self.ref_width_ns  # Waveform units: p.e.
        )
        return sampled

    @classmethod
    def from_camera_readout(cls, readout, gain_channel="ALL"):
        """Create class from a `ctapipe.instrument.CameraReadout`.

        Parameters
        ----------
        readout : `ctapipe.instrument.CameraReadout`
        gain_channel : str
            The reference pulse gain channel to use.
            Choose between 'HIGH', 'LOW' and 'ALL'.

        Returns
        -------
        WaveformModel

        """
        if gain_channel not in VALID_GAIN_CHANNEL:
            raise ValueError(f"gain_channel must be one of {VALID_GAIN_CHANNEL}")

        ref_pulse_shape = readout.reference_pulse_shape
        if gain_channel != "ALL":
            ref_pulse_shape = ref_pulse_shape[np.newaxis, GainChannel[gain_channel]]

        return cls(
            ref_pulse_shape,
            readout.reference_pulse_sample_width,
            (1 / readout.sampling_rate).to(u.ns),
        )


class ImageModel(metaclass=ABCMeta):
    @abstractmethod
    def pdf(self, x, y):
        """Probability density function."""

    def generate_image(self, camera, intensity=50, nsb_level_pe=20, rng=None):
        """Generate a randomized DL1 shower image.
        For the signal, poisson random numbers are drawn from
        the expected signal distribution for each pixel.
        For the background, for each pixel a poisson random number
        if drawn with mean ``nsb_level_pe``.

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
        if rng is None:
            rng = TOYMODEL_RNG

        expected_signal = self.expected_signal(camera, intensity)

        signal = rng.poisson(expected_signal)
        noise = rng.poisson(nsb_level_pe, size=signal.shape)
        image = (signal + noise) - np.mean(noise)

        return image, signal, noise

    def expected_signal(self, camera, intensity):
        """Expected signal in each pixel for the given camera
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

        """
        pdf = self.pdf(camera.pix_x, camera.pix_y)
        return pdf * intensity * camera.pix_area.value


class Gaussian(ImageModel):
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
        psi : u.Quantity[angle]
            rotation angle about the centroid (0=x-axis)

        Returns
        -------
        a `scipy.stats` object

        """
        self.unit = x.unit
        self.x = x
        self.y = y
        self.width = width
        self.length = length
        self.psi = psi

        aligned_covariance = np.array(
            [
                [self.length.to_value(self.unit) ** 2, 0],
                [0, self.width.to_value(self.unit) ** 2],
            ]
        )
        # rotate by psi angle: C' = R C R+
        rotation = linalg.rotation_matrix_2d(self.psi)
        rotated_covariance = rotation @ aligned_covariance @ rotation.T
        self.dist = multivariate_normal(
            mean=u.Quantity([self.x, self.y]).to_value(self.unit),
            cov=rotated_covariance,
        )

    def pdf(self, x, y):
        """2d probability for photon electrons in the camera plane"""
        X = np.column_stack([x.to_value(self.unit), y.to_value(self.unit)])
        return self.dist.pdf(X)


class SkewedGaussian(ImageModel):
    """A shower image that has a skewness along the major axis."""

    def __init__(self, x, y, length, width, psi, skewness):
        """Create 2D skewed Gaussian model for a shower image in a camera.
        Skewness is only applied along the main shower axis.
        See https://en.wikipedia.org/wiki/Skew_normal_distribution and
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skewnorm.html
        for details.

        Parameters
        ----------
        centroid : u.Quantity[length, shape=(2, )]
            position of the centroid of the shower in camera coordinates
        width: u.Quantity[length]
            width of shower (minor axis)
        length: u.Quantity[length]
            length of shower (major axis)
        psi : u.Quantity[angle]
            rotation angle about the centroid (0=x-axis)

        Returns
        -------
        a `scipy.stats` object

        """
        self.unit = x.unit
        self.x = x
        self.y = y
        self.width = width
        self.length = length
        self.psi = psi
        self.skewness = skewness

        a, loc, scale = self._moments_to_parameters()
        self.long_dist = skewnorm(a=a, loc=loc, scale=scale)
        self.trans_dist = norm(loc=0, scale=self.width.to_value(self.unit))
        self.rotation = linalg.rotation_matrix_2d(-self.psi)
        self.mu = u.Quantity([self.x, self.y]).to_value(self.unit)

    def _moments_to_parameters(self):
        """Returns loc and scale from mean, std and skewnewss."""
        # see https://en.wikipedia.org/wiki/Skew_normal_distribution#Estimation
        skew23 = np.abs(self.skewness) ** (2 / 3)
        delta = np.sign(self.skewness) * np.sqrt(
            (np.pi / 2 * skew23) / (skew23 + (0.5 * (4 - np.pi)) ** (2 / 3))
        )
        a = delta / np.sqrt(1 - delta**2)
        scale = self.length.to_value(self.unit) / np.sqrt(1 - 2 * delta**2 / np.pi)
        loc = -scale * delta * np.sqrt(2 / np.pi)

        return a, loc, scale

    def pdf(self, x, y):
        """2d probability for photon electrons in the camera plane."""
        pos = np.column_stack([x.to_value(self.unit), y.to_value(self.unit)])
        long, trans = self.rotation @ (pos - self.mu).T
        return self.trans_dist.pdf(trans) * self.long_dist.pdf(long)


class RingGaussian(ImageModel):
    """A shower image consisting of a ring with gaussian radial profile.

    Simplified model for a muon ring.
    """

    def __init__(self, x, y, radius, sigma):
        self.unit = x.unit
        self.x = x
        self.y = y
        self.sigma = sigma
        self.radius = radius
        self.dist = norm(
            self.radius.to_value(self.unit), self.sigma.to_value(self.unit)
        )

    def pdf(self, x, y):
        """2d probability for photon electrons in the camera plane."""
        r = np.sqrt((x - self.x) ** 2 + (y - self.y) ** 2)
        return self.dist.pdf(r)
