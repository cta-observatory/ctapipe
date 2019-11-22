import numpy as np
from scipy.optimize import minimize
from iminuit import Minuit
import scipy.constants as const
from scipy.stats import norm
from astropy.units import Quantity

__all__ = [
    'kundu_chaudhuri_circle_fit',
    'taubin_circle_fit',
    'psf_likelihood_fit',
    'impact_parameter_chisq_fit',
    'mirror_integration_distance',
    'expected_pixel_light_content',
    'radial_light_intensity',
    'efficiency_fit',
]


def cherenkov_integral(lambda1, lambda2):
    """ integral of int_lambda1^lambda2 lambda^-2 dlambda """
    return 1 / lambda1 - 1 / lambda2


def kundu_chaudhuri_circle_fit(x, y, weights):
    """
    Fast and reliable analytical circle fitting method previously used
    in the H.E.S.S. experiment for muon identification

    Implementation based on [chaudhuri93]_

    Parameters
    ----------
    x: array-like or astropy quantity
        x coordinates of the points
    y: array-like or astropy quantity
        y coordinates of the points
    weights: array-like
        weights of the points

    """

    weights_sum = np.sum(weights)
    mean_x = np.sum(x * weights) / weights_sum
    mean_y = np.sum(y * weights) / weights_sum

    a1 = np.sum(weights * (x - mean_x) * x)
    a2 = np.sum(weights * (y - mean_y) * x)

    b1 = np.sum(weights * (x - mean_x) * y)
    b2 = np.sum(weights * (y - mean_y) * y)

    c1 = 0.5 * np.sum(weights * (x - mean_x) * (x**2 + y**2))
    c2 = 0.5 * np.sum(weights * (y - mean_y) * (x**2 + y**2))

    center_x = (b2 * c1 - b1 * c2) / (a1 * b2 - a2 * b1)
    center_y = (a2 * c1 - a1 * c2) / (a2 * b1 - a1 * b2)

    radius = np.sqrt(np.sum(
        weights * ((center_x - x)**2 + (center_y - y)**2)
    ) / weights_sum)

    return radius, center_x, center_y


def _psf_neg_log_likelihood(params, x, y, weights):
    """
    Negative log-likelihood for a gaussian ring profile

    Parameters
    ----------
    params: 4-tuple
        the fit parameters: (radius, center_x, center_y, std)
    x: array-like
        x coordinates
    y: array-like
        y coordinates
    weights: array-like
        weights for the (x, y) points

    This will usually be x and y coordinates and pe charges of camera pixels
    """
    radius, center_x, center_y, sigma = params
    pixel_distance = np.sqrt((center_x - x)**2 + (center_y - y)**2)

    return np.sum(
        (np.log(sigma) + 0.5 * ((pixel_distance - radius) / sigma)**2) * weights
    )

def all_to_value(*args, unit):
    '''strips unit if all args are convertible to the same unit.

    - does not copy the data
    - the unit returned, will be the one of the 1st arg
    - makes sure all args are convertible to the same unit
    - return the values of all args and the unit
    - Raise a meaningful error in case the args are not of a convertible unit.

    Returns: (*args_without_unit, unit)
    '''
    return tuple(
        Quantity(arg, copy=False).to_value(unit)
        for arg in args
    )



def psf_likelihood_fit(x, y, weights):
    """
    Do a likelihood fit using a ring with gaussian profile.
    Uses the kundu_chaudhuri_circle_fit for the initial guess



    Parameters
    ----------
    x: array-like or astropy quantity
        x coordinates of the points
    y: array-like or astropy quantity
        y coordinates of the points
    weights: array-like
        weights of the points

    This will usually be x and y coordinates and pe charges of camera pixels

    Returns
    -------
    radius: astropy-quantity
        radius of the ring
    center_x: astropy-quantity
        x coordinate of the ring center
    center_y: astropy-quantity
        y coordinate of the ring center
    std: astropy-quantity
        standard deviation of the gaussian profile (indictor for the ring width)
    """
    unit = x.unit
    x, y= all_to_value(x, y, unit=unit)

    start_r, start_x, start_y = kundu_chaudhuri_circle_fit(x, y, weights)

    result = minimize(
        _psf_neg_log_likelihood,
        x0=(start_r, start_x, start_y, 5e-3),
        args=(x, y, weights),
        method='L-BFGS-B',
        bounds=[
            (0, None),      # radius should be positive
            (None, None),
            (None, None),
            (0, None),      # std should be positive
        ],
    )

    if not result.success:
        result.x = np.full_like(result.x, np.nan)

    return result.x * unit


def impact_parameter_chisq_fit(
        pixel_x,
        pixel_y,
        weights,
        radius,
        center_x,
        center_y,
        mirror_radius,
        bins=30,
):
    """
    Impact parameter calculation for a ring fit before.
    This is fitting the theoretical angular light distribution to the
    observed binned angular light distribution using least squares

    Parameters
    ----------
    pixel_x: array-like
        x coordinates of the pixels
    pixel_y: array-like
        y coordinates of the pixels
    weights: array-like
        the weights for the pixel, usually this should be the pe_charge
    radius: float
        ring radius, for example estimated by psf_likelihood_fit
    center_x: float
        x coordinate of the ring center, for example estimated by psf_likelihood_fit
    center_y: float
        y coordinate of the ring center, for example estimated by psf_likelihood_fit
    mirror_radius: float
        the radius of the telescope mirror (circle approximation)
    bins: int
        how many bins to use for the binned fit
    """

    phi = np.arctan2(pixel_y - center_y, pixel_x - center_x)
    hist, edges = np.histogram(phi, bins=bins, range=[-np.pi, np.pi], weights=weights)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    result = minimize(
        _impact_parameter_chisq,
        x0=(mirror_radius / 2, bin_centers[np.argmax(hist)], 1),
        args=(bin_centers, hist, mirror_radius),
        method='L-BFGS-B',
        bounds=[
            (0, None),         # impact parameter should be positive
            (-np.pi, np.pi),   # orientation angle should be in -pi to pi
            (0, None),         # scale should be positive
        ],
    )

    if not result.success:
        result.x = np.full_like(result.x, np.nan)

    imp_par, phi_max, scale = result.x

    return imp_par, phi_max


def mirror_integration_distance(phi, phi_max, impact_parameter, mirror_radius):
    """
    Calculate the distance the muon light went across the telescope mirror
    Function (6) from G. Vacanti et. al., Astroparticle Physics 2, 1994, 1-11

    Parameters
    ----------
    phi: float or array-like
        the orientation angle on the ring
    phi_max: float
        position of the light maximum
    impact_parameter: float
        distance of the muon impact point from the mirror center
    mirror_radius: float
        radius of the telescope mirror (circle approximation)

    Returns
    -------
    distance: float or array-like
    """
    phi = phi - phi_max
    ratio = impact_parameter / mirror_radius
    radicant = 1 - ratio**2 * np.sin(phi)**2

    if impact_parameter > mirror_radius:
        distance = np.empty_like(phi)
        mask = np.logical_and(
            phi < np.arcsin(1 / ratio),
            phi > -np.arcsin(1 / ratio)
        )
        distance[np.logical_not(mask)] = 0
        distance[mask] = 2 * mirror_radius * np.sqrt(radicant[mask])
    else:
        distance = 2 * mirror_radius * (np.sqrt(radicant) + ratio * np.cos(phi))

    return distance


def radial_light_intensity(
        phi,
        phi_max,
        cherenkov_angle,
        impact_parameter,
        pixel_fov,
        mirror_radius,
        lambda1=300e-9,
        lambda2=900e-9,
):
    """
    Amount of photons per azimuthal angle phi on the muon ring as given in
    formula (5) of [vacanti94]_

    Parameters
    ----------
    phi: float or array-like
        the orientation angle on the ring
    phi_max: float
        position of the light maximum
    impact_parameter: float
        distance of the muon impact point from the mirror center
    pixel_fov: float
        field of view of the camera pixels in radian
    mirror_radius: float
        radius of the telescope mirror (circle approximation)
    lambda1: float
        lower integration limit over the cherenkov spectrum in meters
    lambda2: float
        upper integration limit over the cherenkov spectrum in meters

    Returns
    -------
    light_density: float or array-like

    """

    return (
        0.5 * const.fine_structure *
        cherenkov_integral(lambda1, lambda2) *
        pixel_fov / cherenkov_angle *
        np.sin(2 * cherenkov_angle) *
        mirror_integration_distance(phi, phi_max, impact_parameter, mirror_radius)
    )


def expected_pixel_light_content(
        pixel_x,
        pixel_y,
        center_x,
        center_y,
        phi_max,
        cherenkov_angle,
        impact_parameter,
        sigma_psf,
        pixel_fov,
        pixel_diameter,
        mirror_radius,
        focal_length,
        lambda1=300e-9,
        lambda2=900e-9,
):
    """
    Calculate the expected light content of each pixel for a muon ring with the
    given properties

    Parameters
    ----------
    pixel_x: array-like
        x coordinates of the pixels
    pixel_y: array-like
        y coordinates of the pixels
    center_x: float
        x coordinate of the ring center, for example estimated by psf_likelihood_fit
    center_y: float
        y coordinate of the ring center, for example estimated by psf_likelihood_fit
    phi_max: float
        position of the light maximum
    cherenkov_angle: float
        cherenkov_angle of the muon light
    impact_parameter: float
        distance of the muon impact point from the mirror center
    sigma_pdf: float
        standard deviation for the gaussian profile of the ring
    pixel_fov: float
        field of view of the camera pixels in radian
    pixel_diameter: float
        diameter of the pixels
    mirror_radius: float
        radius of the telescope mirror (circle approximation)
    focal_length: float
        focal legth of the telescope
    lambda1: float
        lower integration limit over the cherenkov spectrum in meters
    lambda2: float
        upper integration limit over the cherenkov spectrum in meters

    Returns
    -------
    pe_charge: array-like
        number of photons for each pixel given in pixel_x, pixel_y
    """
    phi = np.arctan2(pixel_y - center_y, pixel_x - center_x)
    pixel_r = np.sqrt((pixel_x - center_x)**2 + (pixel_y - center_y)**2)
    ring_radius = cherenkov_angle * focal_length

    light = radial_light_intensity(
        phi, phi_max,
        cherenkov_angle, impact_parameter,
        pixel_fov, mirror_radius, lambda1, lambda2
    )

    result = light * pixel_diameter * norm.pdf(pixel_r, ring_radius, sigma_psf)
    return result


def efficiency_fit(
        pe_charge,
        pixel_x,
        pixel_y,
        pixel_fov,
        pixel_diameter,
        mirror_radius,
        focal_length,
        lambda1=300e-9,
        lambda2=900e-9,
):
    """
    Estimate optical efficiency for a muon ring using method of [mitchell15]_.
    This is performing several steps:

    1. fit r, x, y and width with the psf_likelihood_fit
    2. fit impact parameter with impact_parameter_chisq_fit
    3. calculate the theoretically expected light contents for each pixel
       for the estimated parameters
    4. calculate the ratio between the observed and the expected number
       of photons.

    Parameters
    ----------
    pe_charge: array-like
        pe_charges of the pixels
    pixel_x: array-like
        x coordinates of the pixels
    pixel_y: array-like
        y coordinates of the pixels
    pixel_fov: float
        field of view of the camera pixels in radian
    pixel_diameter: float
        diameter of the pixels
    mirror_radius: float
        radius of the telescope mirror (circle approximation)
    focal_length: float
        focal legth of the telescope
    lambda1: float
        lower integration limit over the cherenkov spectrum in meters
    lambda2: float
        upper integration limit over the cherenkov spectrum in meters

    Returns
    -------
    radius
    center_x
    center_y
    sigma_psf
    imp_par
    phi_max
    efficiency
    """

    radius, center_x, center_y, sigma_psf = psf_likelihood_fit(
        pixel_x, pixel_y, pe_charge
    )

    imp_par, phi_max = impact_parameter_chisq_fit(
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        weights=pe_charge,
        center_x=center_x,
        center_y=center_y,
        radius=radius,
        mirror_radius=mirror_radius,
        bins=30,
    )

    expected_light = expected_pixel_light_content(
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        center_x=center_x,
        center_y=center_y,
        phi_max=phi_max,
        cherenkov_angle=radius / focal_length,
        impact_parameter=imp_par,
        sigma_psf=sigma_psf,
        pixel_fov=pixel_fov,
        pixel_diameter=pixel_diameter,
        mirror_radius=mirror_radius,
        focal_length=focal_length,
        lambda1=lambda1,
        lambda2=lambda2,
    )

    efficiency = np.sum(pe_charge) / np.sum(expected_light)

    return radius, center_x, center_y, sigma_psf, imp_par, phi_max, efficiency


def _impact_parameter_chisq(params, phi, hist, mirror_radius):
    """ function (6) from G. Vacanti et. al., Astroparticle Physics 2, 1994, 1-11 """

    imp_par, phi_max, scale = params
    theory = mirror_integration_distance(phi, phi_max, imp_par, mirror_radius)

    return np.sum((hist - scale * theory)**2)



def taubin_circle_fit(
    x,
    y,
    mask
):
    """
    reference : Barcelona_Muons_TPA_final.pdf (slide 6)

    Parameters
    ----------
    x: array-like or astropy quantity
        x coordinates of the points
    y: array-like or astropy quantity
        y coordinates of the points
    mask: array-like boolean
        true for pixels surviving the cleaning
    """
    orinal_unit = x.unit
    x, y = all_to_value(x, y, unit=orinal_unit)

    x_masked = x[mask]
    y_masked = y[mask]

    R = x.max()  # x.max() just happens to be identical with R in many cases.

    taubin_r_initial = R / 2
    taubin_error = R * 0.1
    xc = 0
    yc = 0

    # minimization method
    fit = Minuit(
        make_taubin_loss_function(x_masked, y_masked),
        xc=xc,
        yc=yc,
        r=taubin_r_initial,
        error_xc=taubin_error,
        error_yc=taubin_error,
        error_r=taubin_error,
        limit_xc=(-2*R, 2*R),
        limit_yc=(-2*R, 2*R),
        limit_r=(0, R),
        pedantic=False
    )
    fit.migrad()

    radius = fit.values['r'] * orinal_unit
    center_x = fit.values['xc'] * orinal_unit
    center_y = fit.values['yc'] * orinal_unit

    return radius, center_x, center_y


def make_taubin_loss_function(x, y):
    '''closure around taubin_loss_function to make
    surviving pixel positions availaboe inside.

    x, y: positions of pixels surviving the cleaning
        should not be quantities
    '''

    def taubin_loss_function(xc, yc, r):
        """taubin fit formula
        reference : Barcelona_Muons_TPA_final.pdf (slide 6)
        """
        upper_term = (
            (
                (x - xc) ** 2 +
                (y - yc) ** 2
                - r ** 2
            ) ** 2
        ).sum()

        lower_term = (
            (
                (x - xc) ** 2 +
                (y - yc) ** 2
            )
        ).sum()

        return np.abs(upper_term) / np.abs(lower_term)

    return taubin_loss_function
