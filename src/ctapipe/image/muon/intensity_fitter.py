"""
Class for performing a HESS style 2D fit of muon images

To do:
    - Deal with astropy untis better, currently stripped and no checks made
    - unit tests
    - create container class for output

"""
from functools import lru_cache
from math import erf

import numpy as np
from astropy import units as u
from numba import double, vectorize
from scipy.constants import alpha
from scipy.ndimage import correlate1d

from ...containers import MuonEfficiencyContainer
from ...coordinates import TelescopeFrame
from ...core import TelescopeComponent
from ...core.traits import FloatTelescopeParameter, IntTelescopeParameter
from ...exceptions import OptionalDependencyMissing
from ..pixel_likelihood import neg_log_likelihood_approx

try:
    from iminuit import Minuit
except ModuleNotFoundError:
    Minuit = None

__all__ = [
    "MuonIntensityFitter",
]

# ratio of the areas of the unit circle and a square of side lengths 2
CIRCLE_SQUARE_AREA_RATIO = np.pi / 4

# Sqrt of 2, as it is needed multiple times
SQRT2 = np.sqrt(2)


@vectorize([double(double, double, double)], cache=True)
def chord_length(radius, rho, phi):
    """
    Function for integrating the length of a chord across a circle

    Parameters
    ----------
    radius: float or ndarray
        radius of circle
    rho: float or ndarray
        fractional distance of impact point from circle center
    phi: float or ndarray in radians
        rotation angles to calculate length

    Returns
    -------
    float or ndarray:
        chord length
    """
    chord = 1 - (rho**2 * np.sin(phi) ** 2)
    valid = chord >= 0

    if not valid:
        return 0

    if rho <= 1.0:
        # muon has hit the mirror
        chord = radius * (np.sqrt(chord) + rho * np.cos(phi))
    else:
        # muon did not hit the mirror
        chord = 2 * radius * np.sqrt(chord)

    return chord


def intersect_circle(mirror_radius, r, angle, hole_radius=0):
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
    mirror_length = chord_length(mirror_radius, (r / mirror_radius), angle)

    if hole_radius == 0:
        return mirror_length

    hole_length = chord_length(hole_radius, (r / hole_radius), angle)
    return mirror_length - hole_length


def pixels_on_ring(radius, pixel_diameter):
    """Calculate number of pixels of diameter ``pixel_diameter`` on the circumference
    of a circle with radius ``radius``
    """
    circumference = 2 * np.pi * radius
    n_pixels = u.Quantity(circumference / pixel_diameter)
    return int(n_pixels.to_value(u.dimensionless_unscaled))


@lru_cache(maxsize=1000)
def linspace_two_pi(n_points):
    return np.linspace(-np.pi, np.pi, n_points)


def create_profile(
    mirror_radius,
    hole_radius,
    impact_parameter,
    radius,
    phi,
    pixel_diameter,
    oversampling=3,
):
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
    circumference = 2 * np.pi * radius
    pixels_on_circle = int(circumference / pixel_diameter)

    ang = phi + linspace_two_pi(pixels_on_circle * oversampling)

    length = intersect_circle(mirror_radius, impact_parameter, ang, hole_radius)
    length = correlate1d(length, np.ones(oversampling), mode="wrap", axis=0)
    length /= oversampling

    return ang, length


def image_prediction(
    mirror_radius,
    hole_radius,
    impact_parameter,
    phi,
    center_x,
    center_y,
    radius,
    ring_width,
    pixel_x,
    pixel_y,
    pixel_diameter,
    oversampling=3,
    min_lambda=300 * u.nm,
    max_lambda=600 * u.nm,
):
    """
    Parameters
    ----------
    impact_parameter: quantity[length]
        Impact distance of muon
    center_x: quantity[angle]
        Muon ring center in telescope frame
    center_y: quantity[angle]
        Muon ring center in telescope frame
    radius: quantity[angle]
        Radius of muon ring in telescope frame
    ring_width: quantity[angle]
        Gaussian width of muon ring
    pixel_x: quantity[angle]
        Pixel x coordinate in telescope
    pixel_y: quantity[angle]
        Pixel y coordinate in telescope

    Returns
    -------
    ndarray:
        Predicted signal
    """
    return image_prediction_no_units(
        mirror_radius.to_value(u.m),
        hole_radius.to_value(u.m),
        impact_parameter.to_value(u.m),
        phi.to_value(u.rad),
        center_x.to_value(u.rad),
        center_y.to_value(u.rad),
        radius.to_value(u.rad),
        ring_width.to_value(u.rad),
        pixel_x.to_value(u.rad),
        pixel_y.to_value(u.rad),
        pixel_diameter.to_value(u.rad),
        oversampling=oversampling,
        min_lambda_m=min_lambda.to_value(u.m),
        max_lambda_m=max_lambda.to_value(u.m),
    )


@vectorize([double(double, double, double)], cache=True)
def gaussian_cdf(x, mu, sig):
    """
    Function to compute values of a given gaussians
    cumulative distribution function (cdf)

    Parameters
    ----------
    x: float or ndarray
        point, at which the cdf should be evaluated
    mu: float or ndarray
        gaussian mean
    sig: float or ndarray
        gaussian standard deviation

    Returns
    -------
    float or ndarray: cdf-value at x
    """
    return 0.5 * (1 + erf((x - mu) / (SQRT2 * sig)))


def image_prediction_no_units(
    mirror_radius_m,
    hole_radius_m,
    impact_parameter_m,
    phi_rad,
    center_x_rad,
    center_y_rad,
    radius_rad,
    ring_width_rad,
    pixel_x_rad,
    pixel_y_rad,
    pixel_diameter_rad,
    oversampling=3,
    min_lambda_m=300e-9,
    max_lambda_m=600e-9,
):
    """
    Unit-less version of `image_prediction`.
    """

    # First produce angular position of each pixel w.r.t muon center
    dx = pixel_x_rad - center_x_rad
    dy = pixel_y_rad - center_y_rad
    ang = np.arctan2(dy, dx)
    # Add muon rotation angle
    ang += phi_rad

    # Produce smoothed muon profile
    ang_prof, profile = create_profile(
        mirror_radius_m,
        hole_radius_m,
        impact_parameter_m,
        radius_rad,
        phi_rad,
        pixel_diameter_rad,
        oversampling=oversampling,
    )

    # Produce gaussian weight for each pixel given ring width
    radial_dist = np.sqrt(dx**2 + dy**2)
    # The weight is the integral of the ring's radial gaussian profile inside the
    # ring's width
    delta = pixel_diameter_rad / 2
    cdfs = gaussian_cdf(
        [radial_dist + delta, radial_dist - delta], radius_rad, ring_width_rad
    )
    gauss = cdfs[0] - cdfs[1]

    # interpolate profile to find prediction for each pixel
    pred = np.interp(ang, ang_prof, profile)

    # Multiply by integrated emissivity between 300 and 600 nm, and rest of factors to
    # get total number of photons per pixel
    # ^ would be per radian, but no need to put it here, would anyway cancel out below

    pred *= alpha * (min_lambda_m**-1 - max_lambda_m**-1)
    pred *= pixel_diameter_rad / radius_rad
    # multiply by angle (in radians) subtended by pixel width as seen from ring center

    pred *= 0.5 * np.sin(2 * radius_rad)

    # multiply by gaussian weight, to account for "fraction of muon ring" which falls
    # within the pixel
    pred *= gauss

    # Now it would be the total light in an area S delimited by: two radii of the
    # ring, tangent to the sides of the pixel in question, and two circles concentric
    # with the ring, also tangent to the sides of the pixel.
    # A rough correction, assuming pixel is round, is introduced here:
    # [pi*(pixel_diameter/2)**2]/ S. Actually, for the large rings (relative to pixel
    # size) we are concerned with, a good enough approximation is the ratio between a
    # circle's area and that of the square whose side is equal to the circle's
    # diameter. In any case, since in the end we do a data-MC comparison of the muon
    # ring analysis outputs, it is not critical that this value is exact.

    pred *= CIRCLE_SQUARE_AREA_RATIO

    return pred


def build_negative_log_likelihood(
    image,
    optics,
    geometry_tel_frame,
    mask,
    oversampling,
    min_lambda,
    max_lambda,
    spe_width,
    pedestal,
    hole_radius=0 * u.m,
):
    """Create an efficient negative log_likelihood function that does
    not rely on astropy units internally by defining needed values as closures
    in this function.

    The likelihood is the gaussian approximation,
    i.e. the unnumbered equation on page 22 between (24) and (25), from [denaurois2009]_

    The logarithm of the likelihood is calculated analytically as far as possible
    and terms constant under differentation are discarded.
    """

    # get all the needed values and transform them into appropriate units
    mirror_area = optics.mirror_area.to_value(u.m**2)
    mirror_radius = np.sqrt(mirror_area / np.pi)

    # Use only a subset of pixels, indicated by mask:
    pixel_x = geometry_tel_frame.pix_x.to_value(u.rad)
    pixel_y = geometry_tel_frame.pix_y.to_value(u.rad)

    if mask is not None:
        pixel_x = pixel_x[mask]
        pixel_y = pixel_y[mask]
        image = image[mask]
        pedestal = pedestal[mask]

    pixel_diameter = geometry_tel_frame.pixel_width[0].to_value(u.rad)

    min_lambda = min_lambda.to_value(u.m)
    max_lambda = max_lambda.to_value(u.m)

    hole_radius_m = hole_radius.to_value(u.m)

    def negative_log_likelihood(
        impact_parameter,
        phi,
        center_x,
        center_y,
        radius,
        ring_width,
        optical_efficiency_muon,
    ):
        """
        Likelihood function to be called by minimizer

        Parameters
        ----------
        impact_parameter: float
            Impact distance from telescope center
        center_x: float
            center of muon ring in the telescope frame
        center_y: float
            center of muon ring in the telescope frame
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

        # Generate model prediction
        prediction = image_prediction_no_units(
            mirror_radius_m=mirror_radius,
            hole_radius_m=hole_radius_m,
            impact_parameter_m=impact_parameter,
            phi_rad=phi,
            center_x_rad=center_x,
            center_y_rad=center_y,
            radius_rad=radius,
            ring_width_rad=ring_width,
            pixel_x_rad=pixel_x,
            pixel_y_rad=pixel_y,
            pixel_diameter_rad=pixel_diameter,
            oversampling=oversampling,
            min_lambda_m=min_lambda,
            max_lambda_m=max_lambda,
        )

        # scale prediction by optical efficiency of the telescope
        prediction *= optical_efficiency_muon

        return neg_log_likelihood_approx(image, prediction, spe_width, pedestal).sum()

    return negative_log_likelihood


def create_initial_guess(center_x, center_y, radius, telescope_description):
    geometry = telescope_description.camera.geometry
    optics = telescope_description.optics

    focal_length = optics.equivalent_focal_length.to_value(u.m)
    pixel_area = geometry.pix_area[0].to_value(u.m**2)
    pixel_radius = np.sqrt(pixel_area / np.pi) / focal_length

    mirror_radius = np.sqrt(optics.mirror_area.to_value(u.m**2) / np.pi)

    initial_guess = {}
    initial_guess["impact_parameter"] = mirror_radius / 2
    initial_guess["phi"] = 0
    initial_guess["radius"] = radius.to_value(u.rad)
    initial_guess["center_x"] = center_x.to_value(u.rad)
    initial_guess["center_y"] = center_y.to_value(u.rad)
    initial_guess["ring_width"] = 3 * pixel_radius
    initial_guess["optical_efficiency_muon"] = 0.1

    return initial_guess


class MuonIntensityFitter(TelescopeComponent):
    """
    Fit muon ring images with a theoretical model to estimate optical efficiency.

    Function for producing the expected image for a given set of trial
    muon parameters without using astropy units but expecting the input to
    be in the correct ones.

    The image prediction function is currently modeled after :cite:p:`chalmecalvet2013`.

    For more information, also see :cite:p:`muon-review`.
    """

    spe_width = FloatTelescopeParameter(
        help="Width of a single photo electron distribution", default_value=0.5
    ).tag(config=True)

    min_lambda_m = FloatTelescopeParameter(
        help="Minimum wavelength for Cherenkov light in m", default_value=300e-9
    ).tag(config=True)

    max_lambda_m = FloatTelescopeParameter(
        help="Minimum wavelength for Cherenkov light in m", default_value=600e-9
    ).tag(config=True)

    hole_radius_m = FloatTelescopeParameter(
        help="Hole radius of the reflector in m",
        default_value=[
            ("type", "LST_*", 0.308),
            ("type", "MST_*", 0.244),
            ("type", "SST_1M_*", 0.130),
        ],
    ).tag(config=True)

    oversampling = IntTelescopeParameter(
        help="Oversampling for the line integration", default_value=3
    ).tag(config=True)

    def __init__(self, subarray, **kwargs):
        if Minuit is None:
            raise OptionalDependencyMissing("iminuit") from None

        super().__init__(subarray=subarray, **kwargs)
        self._geometries_tel_frame = {
            tel_id: tel.camera.geometry.transform_to(TelescopeFrame())
            for tel_id, tel in subarray.tel.items()
        }

    def __call__(self, tel_id, center_x, center_y, radius, image, pedestal, mask=None):
        """

        Parameters
        ----------
        center_x: Angle quantity
            Initial guess for muon ring center in telescope frame
        center_y: Angle quantity
            Initial guess for muon ring center in telescope frame
        radius: Angle quantity
            Radius of muon ring from circle fitting
        pixel_x: ndarray
            X position of pixels in image from circle fitting
        pixel_y: ndarray
            Y position of pixel in image from circle fitting
        image: ndarray
            Amplitude of image pixels
        mask: ndarray
            mask marking the pixels to be used in the likelihood fit

        Returns
        -------
        MuonEfficiencyContainer
        """
        telescope = self.subarray.tel[tel_id]
        if telescope.optics.n_mirrors != 1:
            raise NotImplementedError(
                "Currently only single mirror telescopes"
                f" are supported in {self.__class__.__name__}"
            )

        negative_log_likelihood = build_negative_log_likelihood(
            image=image,
            optics=telescope.optics,
            geometry_tel_frame=self._geometries_tel_frame[tel_id],
            mask=mask,
            oversampling=self.oversampling.tel[tel_id],
            min_lambda=self.min_lambda_m.tel[tel_id] * u.m,
            max_lambda=self.max_lambda_m.tel[tel_id] * u.m,
            spe_width=self.spe_width.tel[tel_id],
            pedestal=pedestal,
            hole_radius=self.hole_radius_m.tel[tel_id] * u.m,
        )
        negative_log_likelihood.errordef = Minuit.LIKELIHOOD

        initial_guess = create_initial_guess(center_x, center_y, radius, telescope)

        # Create Minuit object with first guesses at parameters
        # strip away the units as Minuit does not like them

        minuit = Minuit(negative_log_likelihood, **initial_guess)

        minuit.print_level = 0

        minuit.errors["impact_parameter"] = 0.5
        minuit.errors["phi"] = np.deg2rad(0.5)
        minuit.errors["ring_width"] = 0.001 * radius.to_value(u.rad)
        minuit.errors["optical_efficiency_muon"] = 0.05

        minuit.limits["impact_parameter"] = (0, None)
        minuit.limits["phi"] = (-np.pi, np.pi)
        minuit.limits["ring_width"] = (0.0, None)
        minuit.limits["optical_efficiency_muon"] = (0.0, None)

        minuit.fixed["radius"] = True
        minuit.fixed["center_x"] = True
        minuit.fixed["center_y"] = True

        # Perform minimisation
        minuit.migrad()

        # Get fitted values
        result = minuit.values

        return MuonEfficiencyContainer(
            impact=result["impact_parameter"] * u.m,
            impact_x=result["impact_parameter"] * np.cos(result["phi"]) * u.m,
            impact_y=result["impact_parameter"] * np.sin(result["phi"]) * u.m,
            width=u.Quantity(np.rad2deg(result["ring_width"]), u.deg),
            optical_efficiency=result["optical_efficiency_muon"],
            is_valid=minuit.valid,
            parameters_at_limit=minuit.fmin.has_parameters_at_limit,
            likelihood_value=minuit.fval,
        )
