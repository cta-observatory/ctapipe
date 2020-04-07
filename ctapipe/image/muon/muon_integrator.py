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
from scipy.stats import norm

from ...containers import MuonIntensityParameter


# ratio of the areas of the unit circle and a square of side lengths 2
CIRCLE_SQUARE_AREA_RATIO = np.pi / 4


__all__ = ['MuonLineIntegrate']


def chord_length(radius, rho, phi):
    """
    Function for integrating the length of a chord across a circle

    Parameters
    ----------
    radius: float
        radius of circle
    rho: float
        fractional distance of impact point from circle center
    phi: ndarray in radians or angle Quantity
        rotation angles to calculate length

    Returns
    -------
    ndarray: chord length
    """
    scalar = np.isscalar(phi)
    phi = u.Quantity(phi, ndmin=1, copy=False)

    chord = 1 - (rho**2 * np.sin(phi)**2)

    if rho <= 1.0:
        # muon has hit the mirror
        chord = radius * (np.sqrt(chord) + rho * np.cos(phi))
    else:
        # muon did not hit the mirror
        chord = 2 * radius * np.sqrt(chord)

    chord[np.isnan(chord)] = 0
    chord[chord < 0] = 0

    if scalar:
        return chord[0]
    return chord


def intersect_circle(mirror_radius, r, angle, hole_radius=0 * u.m):
    """Perform line integration along a given axis in the mirror frame
    given an impact point on the mirror

    Parameters
    ----------
    angle: ndarray
        Angle along which to integrate mirror

    Returns
    --------
    float: length from impact point to mirror edge

    """
    mirror_length = chord_length(
        mirror_radius,
        (r / mirror_radius).to_value(u.dimensionless_unscaled),
        angle
    )

    hole_length = 0 * mirror_length  # .unit
    if hole_radius.value > 0:
        hole_length = chord_length(
            hole_radius,
            (r / hole_radius).to_value(u.dimensionless_unscaled),
            angle
        )

    return mirror_length - hole_length


def pixels_on_ring(radius, pixel_diameter):
    '''Calculate number of pixels of diameter ``pixel_diameter`` on the circumference
    of a circle with radius ``radius``
    '''
    circumference = 2 * np.pi * radius
    n_pixels = u.Quantity(circumference / pixel_diameter)
    return int(n_pixels.to_value(u.dimensionless_unscaled))


def create_profile(mirror_radius, impact_parameter, radius, phi, pixel_width, oversampling=3):
    """
    Perform intersection over all angles and return length

    Parameters
    ----------
    impact_parameter: float
        Impact distance from mirror center
    ang: ndarray
        Angles over which to integrate
    phi: float
        Rotation angle of muon image

    Returns
    -------
    ndarray
        Chord length for each angle
    """

    phi = u.Quantity(phi, u.rad, copy=False)

    circumference = 2 * np.pi * radius
    pixels_on_circle = int(u.Quantity(
        circumference / pixel_width, copy=False
    ).to_value(u.dimensionless_unscaled))

    ang = phi + u.Quantity(np.linspace(-np.pi, np.pi, pixels_on_circle * oversampling), u.rad, copy=False)

    length = intersect_circle(mirror_radius, impact_parameter, ang)
    length = correlate1d(length, np.ones(oversampling), mode='wrap', axis=0)
    length /= oversampling

    return ang, length


def image_prediction(
    mirror_radius,
    impact_parameter,
    phi,
    center_x,
    center_y,
    radius,
    ring_width,
    pixel_x,
    pixel_y,
    pixel_width,
    oversampling=3,
    min_lambda=300 * u.nm,
    max_lambda=600 * u.nm
):
    """Function for producing the expected image for a given set of trial
    muon parameters

    Parameters
    ----------
    impact_parameter: float
        Impact distance of muon
    center_x: float
        Muon ring center in field of view
    center_y: float
        Muon ring center in field of view
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

    # First produce angular position of each pixel w.r.t muon center
    ang = np.arctan2(pixel_y - center_y, pixel_x - center_x)
    # Add muon rotation angle
    ang += phi

    # Produce smoothed muon profile
    ang_prof, profile = create_profile(
        mirror_radius, impact_parameter, radius, phi, pixel_width, oversampling=oversampling,
    )

    # Produce gaussian weight for each pixel given ring width
    radial_dist = np.sqrt((pixel_x - center_x)**2 + (pixel_y - center_y)**2)
    # The weight is the integral of the ring's radial gaussian profile inside the
    # ring's width
    delta = pixel_width / 2
    cdfs = norm.cdf([radial_dist + delta, radial_dist - delta], radius, ring_width)
    gauss = cdfs[0] - cdfs[1]

    # interpolate profile to find prediction for each pixel
    pred = u.Quantity(np.interp(ang, ang_prof, profile), u.m, copy=False)

    # Multiply by integrated emissivity between 300 and 600 nm, and rest of factors to
    # get total number of photons per pixel
    # ^ would be per radian, but no need to put it here, would anyway cancel out below

    pred *= alpha * (min_lambda**-1 - max_lambda**-1)
    pred *= (pixel_width / radius).to_value(u.dimensionless_unscaled)
    # multiply by angle (in radians) subtended by pixel width as seen from ring center

    pred *= np.sin(2 * np.deg2rad(radius))

    # multiply by gaussian weight, to account for "fraction of muon ring" which falls
    # within the pixel
    pred *= gauss

    # Now it would be the total light in an area S delimited by: two radii of the
    # ring, tangent to the sides of the pixel in question, and two circles concentric
    # with the ring, also tangent to the sides of the pixel.
    # A rough correction, assuming pixel is round, is introduced here:
    # [pi*(pixel_width/2)**2]/ S. Actually, for the large rings (relative to pixel
    # size) we are concerned with, a good enough approximation is the ratio between a
    # circle's area and that of the square whose side is equal to the circle's
    # diameter. In any case, since in the end we do a data-MC comparison of the muon
    # ring analysis outputs, it is not critical that this value is exact.
    pred *= CIRCLE_SQUARE_AREA_RATIO

    return pred.to_value(u.dimensionless_unscaled)


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

    def __init__(
        self,
        mirror_radius,
        hole_radius,
        pixel_width=0.2,
        oversample_bins=3,
        sct_flag=False,
        secondary_radius=1.
    ):

        self.mirror_radius = mirror_radius
        self.hole_radius = hole_radius
        self.pixel_width = pixel_width
        self.oversample_bins = oversample_bins
        self.sct_flag = sct_flag
        self.secondary_radius = secondary_radius
        self.pixel_x = 0
        self.pixel_y = 0
        self.image = 0
        self.prediction = 0
        self.minlambda = 300.e-9 * u.m
        self.maxlambda = 600.e-9 * u.m
        self.unit = u.deg

    def likelihood(self, impact_parameter, phi, center_x, center_y,
                   radius, ring_width, optical_efficiency_muon):
        """
        Likelihood function to be called by minimiser

        Parameters
        ----------
        impact_parameter: float
            Impact distance from telescope center
        center_x: float
            center of muon ring in FoV
        center_y: float
            center of muon ring in FoV
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
        # center_x *= self.unit
        # center_y *= self.unit
        # radius *= self.unit
        # ring_width *= self.unit
        # impact_parameter *= u.m
        # phi *= u.rad

        # Generate model prediction
        self.prediction = self.image_prediction(
            impact_parameter,
            phi,
            center_x,
            center_y,
            radius,
            ring_width,
            self.pixel_x.value,
            self.pixel_y.value,
        )

        # TEST: extra scaling factor, HESS style (ang pix size /2piR) # NOTE: not needed!
        # (after changes introduced in 20200331)
        # scalenpix = self.pixel_width / (2.*np.pi * radius)
        # self.prediction *= scalenpix.value

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

        sq = 1 / np.sqrt(2 * np.pi * (ped**2 + pred * (1 + spe_width**2) ))
        diff = (image - pred)**2
        denom = 2 * (ped**2 + pred * (1 + spe_width**2) )
        expo = np.exp(-diff / denom) * u.m
        sm = expo < 1e-300 * u.m
        expo[sm] = 1e-300 * u.m

        log_value = sq * expo / u.m

        likelihood_value = -2 * np.log(log_value)

        return likelihood_value

    def fit_muon(self, center_x, center_y, radius, pixel_x, pixel_y, image):
        """

        Parameters
        ----------
        center_x: float
            center of muon ring in the field of view from circle fitting
        center_y: float
            center of muon ring in the field of view from circle fitting
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
        center_x.to(u.deg)
        center_y.to(u.deg)

        # Return interesting stuff
        fitoutput = MuonIntensityParameter()

        init_params = {}
        init_errs = {}
        init_constrain = {}
        init_params['impact_parameter'] = 4.
        init_params['phi'] = 0.
        init_params['radius'] = radius.value
        init_params['center_x'] = center_x.value
        init_params['center_y'] = center_y.value
        init_params['ring_width'] = 0.1
        init_params['optical_efficiency_muon'] = 0.1
        init_errs['error_impact_parameter'] = 2.
        init_constrain['limit_impact_parameter'] = (0., 25.)
        init_errs['error_phi'] = 0.1
        init_errs['error_ring_width'] = 0.001 * radius.value
        init_errs['error_optical_efficiency_muon'] = 0.05
        init_constrain['limit_phi'] = (-np.pi, np.pi)
        init_constrain['fix_radius'] = True
        init_constrain['fix_center_x'] = True
        init_constrain['fix_center_y'] = True
        init_constrain['limit_ring_width'] = (0., 1.)
        # init_constrain['limit_optical_efficiency_muon'] = (0., 1.)
        # ^Unneeded constraint - and misleading in case of changes leading to >1 values!

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
