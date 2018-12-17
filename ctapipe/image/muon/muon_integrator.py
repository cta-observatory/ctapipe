"""
Class for performing a HESS style 2D fit of muon images

To do:
    - Deal with astropy untis better, currently stripped and no checks made
    - unit tests
    - create container class for output

"""
import numpy as np
from scipy.ndimage.filters import correlate1d
from iminuit import Minuit
from astropy import units as u
from astropy.constants import alpha
from ...io.containers import MuonIntensityParameter
from scipy.stats import norm

import logging

__all__ = ['MuonLineIntegrate']

logger = logging.getLogger(__name__)


class MuonLineIntegrate:
    """
    Object for calculating the expected 2D shape of muon image for a given
    mirror geometry. Geometry is passed to the class as
    a series of points defining the outer edge of the array
    and a set of points defining the shape of the central hole (if present).
    Muon profiles are then calculated bsed of the line integral along a given
    axis from the muon impact point.

    Expected 2D images can then be generated when the pixel geometry
    is passed to the class.

    Parameters
    ----------
    mirror_radius: float
        Radius of telescope mirror (circular approx)
    hole_radius: float
        Radius of telescope mirror hole (circular approx)
    pixel_width: float
        width of pixel in camera
    oversample_bins: int
        number of angular bins to evaluate for each pixel width
    sct_flag: bool
        flags whether the telescope uses Schwarzschild-Couder Optics
    secondary_radius: float
        Radius of the secondary mirror (circular approx) for dual
        mirror telescopes
    minlambda: float
        minimum wavelength for integration (300nm typical)
    maxlambda: float
        maximum wavelength for integration (600nm typical)
    photemit: float
        1/lambda^2 integrated over the above defined wavelength range
        multiplied by the fine structure constant (1/137)
    """

    def __init__(self, mirror_radius, hole_radius, pixel_width=0.2,
                 oversample_bins=3, sct_flag=False, secondary_radius=1.):

        self.mirror_radius = mirror_radius
        self.hole_radius = hole_radius
        self.pixel_width = pixel_width
        self.oversample_bins = oversample_bins
        self.sct_flag = sct_flag
        self.secondary_radius = secondary_radius
        self.pixel_x = 0
        self.pixel_y = 0
        self.image = 0 * u.deg
        self.prediction = 0
        self.minlambda = 300.e-9 * u.m
        self.maxlambda = 600.e-9 * u.m
        self.photemit = alpha * (self.minlambda**-1 -
                                 self.maxlambda**-1)  # 12165.45
        self.unit = u.deg

    @staticmethod
    def chord_length(radius, rho, phi):
        """
        Function for integrating the length of a chord across a circle

        Parameters
        ----------
        radius: float
            radius of circle
        rho: float
            fractional distance of impact point from array centre
        phi: ndarray
            rotation angles to calculate length

        Returns
        -------
        ndarray: chord length
        """
        chord = 1 - (rho * rho * np.sin(phi) * np.sin(phi))
        if rho <= 1.0:
            chord = radius * (np.sqrt(chord) + rho * np.cos(phi))
        elif rho > 1.0:
            chord = 2. * radius * np.sqrt(chord)

        chord[np.isnan(chord)] = 0
        chord[chord < 0] = 0

        return chord

    def intersect_circle(self, r, angle):
        """Perform line integration along a given axis in the mirror frame
        given an impact point on the mirrot

        Parameters
        ----------
        impact_x: float
            Impact position on mirror (tilted telescope system)
        impact_y: float
            Impact position on mirror (tilted telescope system)
        angle: float
            Angle along which to integrate mirror

        Returns
        --------
        float: length from impact point to mirror edge

        """
        mirror_length = self.chord_length(
            self.mirror_radius, r / self.mirror_radius.value, angle
        )
        hole_length = 0 * mirror_length  # .unit
        if self.hole_radius > 0:
            hole_length = self.chord_length(
                self.hole_radius, r / self.hole_radius.value, angle
            )

        if self.sct_flag:
            self.sct_flag = False  # Do not treat differently for now...work in progress

        if self.sct_flag:
            secondary_length = self.chord_length(
                self.secondary_radius, r / self.secondary_radius, angle
            )
            # Should be areas not lengths here?
            factor = mirror_length - (secondary_length)
            factor /= (mirror_length - hole_length)

        if not self.sct_flag:
            return mirror_length - hole_length
        else:
            return (mirror_length - hole_length + secondary_length)

    def plot_pos(self, impact_parameter, radius, phi):
        """
        Perform intersection over all angles and return length

        Parameters
        ----------
        impact_parameter: float
            Impact distance from mirror centre
        ang: ndarray
            Angles over which to integrate
        phi: float
            Rotation angle of muon image

        Returns
        -------
        ndarray
            Chord length for each angle
        """

        bins = int((2 * np.pi * radius) / self.pixel_width.value) * self.oversample_bins
        # ang = np.linspace(-np.pi * u.rad + phi, np.pi * u.rad + phi, bins)
        ang = np.linspace(-np.pi + phi, np.pi + phi, bins)
        l = self.intersect_circle(impact_parameter, ang)
        l = correlate1d(l, np.ones(self.oversample_bins), mode='wrap', axis=0)
        l /= self.oversample_bins

        return ang, l

    def pos_to_angle(self, centre_x, centre_y, pixel_x, pixel_y):
        """
        Convert pixel positions from x,y coordinates to rotation angle

        Parameters
        ----------
        centre_x: float
            Reconstructed image centre
        centre_y: float
            Reconstructed image centre
        pixel_x: ndarray
            Pixel x position
        pixel_y: ndarray
            Pixel y position

        Returns
        -------
         ndarray:
            Pixel rotation angle

        """
        del_x = pixel_x - centre_x
        del_y = pixel_y - centre_y

        ang = np.arctan2(del_x, del_y)
        return ang

    def image_prediction(self, impact_parameter, phi, centre_x,
                         centre_y, radius, ring_width, pixel_x, pixel_y, ):
        """Function for producing the expected image for a given set of trial
        muon parameters

        Parameters
        ----------
        impact_parameter: float
            Impact distance of muon
        centre_x: float
            Muon ring centre in field of view
        centre_y: float
            Muon ring centre in field of view
        radius: float
            Radius of muon ring
        ring_width: float
            Gaussian width of muon ring
        pixel_x: ndarray
            Pixel x coordinate
        pixel_y: ndarray
            Pixel y coordinate

        Returns
        -------
        ndarray:
            Predicted signal

        """

        # First produce angular position of each pixel w.r.t muon centre
        ang = self.pos_to_angle(centre_x, centre_y, pixel_x, pixel_y)
        # Add muon rotation angle
        ang += phi
        # Produce smoothed muon profile

        ang_prof, profile = self.plot_pos(impact_parameter, radius, phi)
        # Produce gaussian weight for each pixel give ring width
        radial_dist = np.sqrt((pixel_x - centre_x)**2 + (pixel_y - centre_y)**2)
        gauss = norm.pdf(radial_dist, radius, ring_width)

        # interpolate profile to find prediction for each pixel
        pred = np.interp(ang, ang_prof, profile) * u.m

        # Multiply by integrated emissivity between 300 and 600 nm
        photval = self.photemit / u.deg
        pred *= 0.5 * photval

        # weight by pixel width
        pred *= (self.pixel_width.value / radius)
        pred *= np.sin(2 * radius)
        # weight by gaussian width
        pred *= self.pixel_width.value * gauss

        return pred

    def likelihood(self, impact_parameter, phi, centre_x, centre_y,
                   radius, ring_width, optical_efficiency_muon):
        """
        Likelihood function to be called by minimiser

        Parameters
        ----------
        impact_parameter: float
            Impact distance from telescope centre
        centre_x: float
            Centre of muon ring in FoV
        centre_y: float
            Centre of muon ring in FoV
        radius: float
            Radius of muon ring
        ring_width: float
            Gaussian width of muon ring
        optical_efficiency_muon: float
            Efficiency of the optical system

        Returns
        -------
        float: Likelihood that model matches data
        """
        # centre_x *= self.unit
        # centre_y *= self.unit
        # radius *= self.unit
        # ring_width *= self.unit
        # impact_parameter *= u.m
        # phi *= u.rad

        # Generate model prediction
        self.prediction = self.image_prediction(
            impact_parameter,
            phi,
            centre_x,
            centre_y,
            radius,
            ring_width,
            self.pixel_x.value,
            self.pixel_y.value,
        )
        # TEST: extra scaling factor, HESS style (ang pix size /2piR)

        scalenpix = self.pixel_width / (2.*np.pi * radius)
        self.prediction *= scalenpix.value

        # scale prediction by optical efficiency of array
        self.prediction *= optical_efficiency_muon

        # Multiply sum of likelihoods by -2 to make them behave like chi-squared
        like_value = np.sum(self.calc_likelihood(self.image, self.prediction, 0.5, 1.1))

        return like_value

    @staticmethod
    def calc_likelihood(image, pred, spe_width, ped):
        """Calculate likelihood of prediction given the measured signal,
        gaussian approx from [denaurois2009]_

        Parameters
        ----------
        image: ndarray
            Pixel amplitudes from image
        pred: ndarray
            Predicted pixel amplitudes from model
        spe_width: ndarray
            width of single p.e. distributio
        ped: ndarray
            width of pedestal

        Returns
        -------
        ndarray: likelihood for each pixel

        """
        #ped = ped * u.deg
        #image = image * u.deg
        pred = pred * u.deg
        sq = 1 / np.sqrt(2 * np.pi * (ped**2 + pred * (1 + spe_width**2) ))
        diff = (image - pred)**2
        denom = 2 * (ped**2 + pred * (1 + spe_width**2) )
        expo = np.exp(-diff / denom) * u.m
        sm = expo < 1e-300 * u.m
        expo[sm] = 1e-300 * u.m

        log_value = sq * expo / u.m

        likelihood_value = -2 * np.log(log_value)

        return likelihood_value

    def fit_muon(self, centre_x, centre_y, radius, pixel_x, pixel_y, image):
        """

        Parameters
        ----------
        centre_x: float
            Centre of muon ring in the field of view from circle fitting
        centre_y: float
            Centre of muon ring in the field of view from circle fitting
        radius: float
            Radius of muon ring from circle fitting
        pixel_x: ndarray
            X position of pixels in image from circle fitting
        pixel_y: ndarray
            Y position of pixel in image from circle fitting
        image: ndarray
            Amplitude of image pixels

        Returns
        -------
        MuonIntensityParameters
        """

        # First store these parameters in the class so we can use them in minimisation
        self.image = image
        self.pixel_x = pixel_x.to(u.deg)
        self.pixel_y = pixel_y.to(u.deg)
        self.unit = pixel_x.unit

        radius.to(u.deg)
        centre_x.to(u.deg)
        centre_y.to(u.deg)

        # Return interesting stuff
        fitoutput = MuonIntensityParameter()

        init_params = {}
        init_errs = {}
        init_constrain = {}
        init_params['impact_parameter'] = 4.
        init_params['phi'] = 0.
        init_params['radius'] = radius.value
        init_params['centre_x'] = centre_x.value
        init_params['centre_y'] = centre_y.value
        init_params['ring_width'] = 0.1
        init_params['optical_efficiency_muon'] = 0.1
        init_errs['error_impact_parameter'] = 2.
        init_constrain['limit_impact_parameter'] = (0., 25.)
        init_errs['error_phi'] = 0.1
        init_errs['error_ring_width'] = 0.001 * radius.value
        init_errs['error_optical_efficiency_muon'] = 0.05
        init_constrain['limit_phi'] = (-np.pi, np.pi)
        init_constrain['fix_radius'] = True
        init_constrain['fix_centre_x'] = True
        init_constrain['fix_centre_y'] = True
        init_constrain['limit_ring_width'] = (0., 1.)
        init_constrain['limit_optical_efficiency_muon'] = (0., 1.)

        logger.debug("radius = %3.3f pre migrad", radius)

        # Create Minuit object with first guesses at parameters
        # strip away the units as Minuit doesnt like them

        minuit = Minuit(
            self.likelihood,
            # forced_parameters=parameter_names,
            **init_params,
            **init_errs,
            **init_constrain,
            errordef=0.1,
            print_level=0,
            pedantic=False
        )

        # Perform minimisation
        minuit.migrad()

        # Get fitted values
        fit_params = minuit.values

        fitoutput.impact_parameter = fit_params['impact_parameter'] * u.m
        # fitoutput.phi = fit_params['phi']*u.rad
        fitoutput.impact_parameter_pos_x = fit_params['impact_parameter'] * np.cos(
            fit_params['phi'] * u.rad) * u.m
        fitoutput.impact_parameter_pos_y = fit_params['impact_parameter'] * np.sin(
            fit_params['phi'] * u.rad) * u.m
        fitoutput.ring_width = fit_params['ring_width'] * self.unit
        fitoutput.optical_efficiency_muon = fit_params['optical_efficiency_muon']

        fitoutput.prediction = self.prediction

        return fitoutput
