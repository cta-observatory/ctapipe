# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utilities to generate toymodel (fake) reconstruction inputs for testing
purposes.

Examples:

.. code-block:: python

    >>> from ctapipe.instrument import CameraGeometry
    >>> geom = CameraGeometry.make_rectangular(20, 20)
    >>> showermodel = Gaussian(x=0.25 * u.m, y=0.0 * u.m,
    length=0.1 * u.m, width=0.02 * u.m, psi='40d')
    >>> image, signal, noise = showermodel.generate_image(geom, intensity=1000)
    >>> print(image.shape)
    (400,)
"""
import numpy as np
from ctapipe.utils import linalg
from ctapipe.image.hillas import camera_to_shower_coordinates
import astropy.units as u
from astropy.coordinates import Angle
from scipy.stats import multivariate_normal, skewnorm, norm
from scipy.ndimage import convolve1d
from abc import ABCMeta, abstractmethod

__all__ = [
    "WaveformModel",
    "Gaussian",
    "SkewedGaussian",
    "ImageModel",
    "obtain_time_image",
]


@u.quantity_input(
    x=u.m,
    y=u.m,
    centroid_x=u.m,
    centroid_y=u.m,
    psi=u.deg,
    time_gradient=u.ns / u.m,
    time_intercept=u.ns,
)
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
    time_gradient : u.Quantity[time/length]
        Rate at which the time changes with distance along the shower axis
    time_intercept : u.Quantity[time]
        Pulse time at the shower centroid

    Returns
    -------
    float or ndarray
        Pulse time in nanoseconds at (x, y)

    """
    longitudinal, _ = camera_to_shower_coordinates(x, y, centroid_x, centroid_y, psi)
    longitudinal_m = longitudinal.to_value(u.m)
    time_gradient_ns_m = time_gradient.to_value(u.ns / u.m)
    time_intercept_ns = time_intercept.to_value(u.ns)
    return longitudinal_m * time_gradient_ns_m + time_intercept_ns


class WaveformModel:
    @u.quantity_input(reference_pulse_sample_width=u.ns, sample_width=u.ns)
    def __init__(self, reference_pulse, reference_pulse_sample_width, sample_width):
        """Generate a toy model waveform using the reference pulse shape of a
        camera. Useful for testing image extraction algorithms.

        Does not include the electronic noise and the Excess Noise Factor of
        the photosensor, and therefore should not be used to make charge
        resolution conclusions about a camera.

        Parameters
        ----------
        reference_pulse_sample_width : u.Quantity[time]
            Sample width of the reference pulse shape
        reference_pulse : ndarray
            Reference pulse shape
        sample_width : u.Quantity[time]
            Sample width of the waveform

        """
        self.upsampling = 10
        reference_pulse_sample_width = reference_pulse_sample_width.to_value(u.ns)
        sample_width_ns = sample_width.to_value(u.ns)
        ref_max_sample = reference_pulse.size * reference_pulse_sample_width
        reference_pulse_x = np.arange(0, ref_max_sample, reference_pulse_sample_width)
        self.ref_width_ns = sample_width_ns / self.upsampling
        self.ref_interp_x = np.arange(0, reference_pulse_x.max(), self.ref_width_ns)
        self.ref_interp_y = np.interp(
            self.ref_interp_x, reference_pulse_x, reference_pulse
        )
        self.ref_interp_y /= self.ref_interp_y.sum() * self.ref_width_ns
        self.origin = self.ref_interp_y.argmax() - self.ref_interp_y.size // 2

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
            Shape (n_pixels, n_samples)

        """
        n_pixels = charge.size
        n_upsampled_samples = n_samples * self.upsampling
        readout = np.zeros((n_pixels, n_upsampled_samples))

        sample = (time / self.ref_width_ns).astype(np.int)
        outofrange = (sample < 0) | (sample >= n_upsampled_samples)
        sample[outofrange] = 0
        charge[outofrange] = 0
        readout[np.arange(n_pixels), sample] = charge
        convolved = convolve1d(
            readout, self.ref_interp_y, mode="constant", origin=self.origin
        )
        sampled = (
            convolved.reshape(
                (n_pixels, convolved.shape[-1] // self.upsampling, self.upsampling)
            ).sum(-1)
            * self.ref_width_ns  # Waveform units: p.e.
        )
        return sampled

    @classmethod
    def from_camera_readout(cls, readout, gain_channel=0):
        """Create class from a `ctapipe.instrument.CameraReadout`.

        Parameters
        ----------
        readout : `ctapipe.instrument.CameraReadout`
        gain_channel : int
            The reference pulse gain channel to use

        Returns
        -------
        WaveformModel

        """
        return cls(
            readout.reference_pulse_shape[gain_channel],
            readout.reference_pulse_sample_width,
            (1 / readout.sampling_rate).to(u.ns),
        )


class ImageModel(metaclass=ABCMeta):
    @u.quantity_input(x=u.m, y=u.m)
    @abstractmethod
    def pdf(self, x, y):
        """Probability density function.
        """

    def generate_image(self, camera, intensity=50, nsb_level_pe=20):
        """Generate a randomized DL1 shower image.
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
        """2d probability for photon electrons in the camera plane"""
        aligned_covariance = np.array(
            [[self.length.to_value(u.m) ** 2, 0], [0, self.width.to_value(u.m) ** 2]]
        )
        # rotate by psi angle: C' = R C R+
        rotation = linalg.rotation_matrix_2d(self.psi)
        rotated_covariance = rotation @ aligned_covariance @ rotation.T

        return multivariate_normal(
            mean=[self.x.to_value(u.m), self.y.to_value(u.m)], cov=rotated_covariance,
        ).pdf(np.column_stack([x.to_value(u.m), y.to_value(u.m)]))


class SkewedGaussian(ImageModel):
    """A shower image that has a skewness along the major axis.
    """

    @u.quantity_input(x=u.m, y=u.m, length=u.m, width=u.m)
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
        """Returns loc and scale from mean, std and skewnewss."""
        # see https://en.wikipedia.org/wiki/Skew_normal_distribution#Estimation
        skew23 = np.abs(self.skewness) ** (2 / 3)
        delta = np.sign(self.skewness) * np.sqrt(
            (np.pi / 2 * skew23) / (skew23 + (0.5 * (4 - np.pi)) ** (2 / 3))
        )
        a = delta / np.sqrt(1 - delta ** 2)
        scale = self.length.to_value(u.m) / np.sqrt(1 - 2 * delta ** 2 / np.pi)
        loc = -scale * delta * np.sqrt(2 / np.pi)

        return a, loc, scale

    @u.quantity_input(x=u.m, y=u.m)
    def pdf(self, x, y):
        """2d probability for photon electrons in the camera plane."""
        mu = u.Quantity([self.x, self.y]).to_value(u.m)

        rotation = linalg.rotation_matrix_2d(-Angle(self.psi))
        pos = np.column_stack([x.to_value(u.m), y.to_value(u.m)])
        long, trans = rotation @ (pos - mu).T

        trans_pdf = norm(loc=0, scale=self.width.to_value(u.m)).pdf(trans)

        a, loc, scale = self._moments_to_parameters()

        return trans_pdf * skewnorm(a=a, loc=loc, scale=scale).pdf(long)


class RingGaussian(ImageModel):
    """A shower image consisting of a ring with gaussian radial profile.
    Simplified model for a muon ring.

    """

    @u.quantity_input(x=u.m, y=u.m, radius=u.m, sigma=u.m)
    def __init__(self, x, y, radius, sigma):
        self.x = x
        self.y = y
        self.sigma = sigma
        self.radius = radius

    @u.quantity_input(x=u.m, y=u.m)
    def pdf(self, x, y):
        """2d probability for photon electrons in the camera plane."""

        r = np.sqrt((x - self.x) ** 2 + (y - self.y) ** 2)

        return norm(self.radius.to_value(u.m), self.sigma.to_value(u.m),).pdf(r)
