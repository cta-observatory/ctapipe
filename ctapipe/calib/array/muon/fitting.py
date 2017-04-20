import numpy as np
from scipy.optimize import minimize
import scipy.constants as const
from scipy.stats import norm

__all__ = [
    'kundu_chaudhuri_circle_fit',
    'psf_likelihood_fit',
    'impact_parameter_chisq_fit',
    'mirror_integration_distance',
    'expected_pixel_light_content',
    'radial_light_intensity',
    'efficiency_fit',
]


def cherenkov_integral(lambda1, lambda2):
    ''' integral of int_lambda1^lambda2 lambda^-2 dlambda '''
    return 1 / lambda1 - 1 / lambda2


def kundu_chaudhuri_circle_fit(x, y, weights):
    '''
    Fast, analytic calculation of circle center and radius from
    x, y and weights. This should be pixel positions and photon equivalents

    Reference:
    B. B. Chaudhuri und P. Kundu.
    "Optimum circular fit to weighted data in multi-dimensional space".
    In: Pattern Recognition Letters 14.1 (1993), S. 1–6
    '''
    # handle astropy units
    try:
        unit = x.unit
        assert x.unit == y.unit
        x = x.value
        y = y.value
    except AttributeError:
        unit = None

    mean_x = np.average(x, weights=weights)
    mean_y = np.average(y, weights=weights)

    a1 = np.sum(weights * (x - mean_x) * x)
    a2 = np.sum(weights * (y - mean_y) * x)

    b1 = np.sum(weights * (x - mean_x) * y)
    b2 = np.sum(weights * (y - mean_y) * y)

    c1 = 0.5 * np.sum(weights * (x - mean_x) * (x**2 + y**2))
    c2 = 0.5 * np.sum(weights * (y - mean_y) * (x**2 + y**2))

    center_x = (b2 * c1 - b1 * c2) / (a1 * b2 - a2 * b1)
    center_y = (a2 * c1 - a1 * c2) / (a2 * b1 - a1 * b2)

    radius = np.sqrt(np.average(
        (center_x - x)**2 + (center_y - y)**2,
        weights=weights,
    ))

    if unit:
        radius *= unit
        center_x *= unit
        center_y *= unit

    return radius, center_x, center_y


def psf_likelihood_fit(x, y, weights):
    ''' Apply a gaussian likelihood for a muon ring
    '''
    def _psf_likelihood_function(params, x, y, weights):
        ''' negative log-likelihood for a gaussian ring profile '''
        radius, center_x, center_y, sigma = params
        pixel_distance = np.sqrt((center_x - x)**2 + (center_y - y)**2)

        return np.sum(
            (np.log(sigma) + 0.5 * ((pixel_distance - radius) / sigma)**2) * weights
        )

    try:
        unit = x.unit
        assert x.unit == y.unit
        x = x.value
        y = y.value
    except AttributeError:
        unit = None

    start_r, start_x, start_y = kundu_chaudhuri_circle_fit(x, y, weights)

    result = minimize(
        _psf_likelihood_function,
        x0=(start_r, start_x, start_y, 5e-3),
        args=(x, y, weights),
        method='Powell',
    )

    if not result.success:
        result.x = np.full_like(result.x, np.nan)

    if unit:
        return result.x * unit

    return result.x


def impact_parameter_chisq_fit(
        pixel_x,
        pixel_y,
        weights,
        center_x,
        center_y,
        radius,
        mirror_radius,
        bins=30,
        ):
    ''' Impact parameter calculation
    '''

    phi = np.arctan2(pixel_y - center_y, pixel_x - center_x)
    hist, edges = np.histogram(phi, bins=bins, range=[-np.pi, np.pi], weights=weights)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    result = minimize(
        _impact_parameter_chisq,
        x0=(mirror_radius / 2, bin_centers[np.argmax(hist)], 1),
        args=(bin_centers, hist, mirror_radius),
        method='L-BFGS-B',
        bounds=[(0, None), (-np.pi, np.pi), (0, None)],
    )

    if not result.success:
        result.x = np.full_like(result.x, np.nan)

    imp_par, phi_max, scale = result.x

    return imp_par, phi_max


def mirror_integration_distance(phi, phi_max, impact_parameter, mirror_radius):
    ''' function (6) from G. Vacanti et. al., Astroparticle Physics 2, 1994, 1-11 '''
    phi = phi - phi_max
    ratio = impact_parameter / mirror_radius
    radicant = 1 - ratio**2 * np.sin(phi)**2

    if impact_parameter > mirror_radius:
        D = np.empty_like(phi)
        mask = np.logical_and(
            phi < np.arcsin(1 / ratio),
            phi > -np.arcsin(1 / ratio)
        )
        D[np.logical_not(mask)] = 0
        D[mask] = 2 * mirror_radius * np.sqrt(radicant[mask])
    else:
        D = 2 * mirror_radius * (np.sqrt(radicant) + ratio * np.cos(phi))

    return D


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
    '''
    Amount of photons per azimuthal angle phi on the muon ring as given in
    G. Vacanti et. al., Astroparticle Physics 2, 1994, 1-11, formula (5)
    '''

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
    '''
    Calculate the expected light content of each pixel for a muon ring with the
    given properties
    '''
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
    '''
    A Generic Algorithm for IACT Optical Efficiency Calibration using Muons,
    Allison Mitchell et al. arXiv: 1509.04258v1
    '''

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


def _impact_parameter_chisq(params,  phi, hist, mirror_radius):
    ''' function (6) from G. Vacanti et. al., Astroparticle Physics 2, 1994, 1-11 '''

    imp_par, phi_max, scale = params
    theory = mirror_integration_distance(phi, phi_max, imp_par, mirror_radius)

    return np.sum((hist - scale * theory)**2)
