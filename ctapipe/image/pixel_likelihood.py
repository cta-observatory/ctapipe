"""
Class for calculation of likelihood of a pixel expectation, given the pixel amplitude,
the level of noise in the pixel and the photoelectron resolution.
This calculation is taken from [denaurois2009]_.

The likelihood is essentially a poissonian convolved with a gaussian, at low signal
a full possonian approach must be adopted, which requires the sum of contibutions
over a number of potential contributing photoelectrons (which is slow).
At high signal this simplifies to a gaussian approximation.

The full and gaussian approximations are implemented, in addition to a general purpose
implementation, which tries to intellegently switch
between the two. Speed tests are below:

neg_log_likelihood_approx(image, prediction, spe, ped)
29.8 µs per loop

neg_log_likelihood_numeric(image, prediction, spe, ped)
93.4 µs per loop

neg_log_likelihood(image, prediction, spe, ped)
59.9 µs per loop

TODO:
=====
- Need to implement more tests, particularly checking for error states
- Additional terms may be useful to add to the likelihood
"""

import numpy as np
from scipy.integrate import quad
from scipy.stats import poisson

__all__ = [
    "neg_log_likelihood_approx",
    "neg_log_likelihood_numeric",
    "neg_log_likelihood",
    "mean_poisson_likelihood_gaussian",
    "mean_poisson_likelihood_full",
    "PixelLikelihoodError",
    "chi_squared",
]


class PixelLikelihoodError(RuntimeError):
    pass


def neg_log_likelihood_approx(image, prediction, spe_width, pedestal):
    """Calculate negative log likelihood for telescope.

    Gaussian approximation from [denaurois2009]_, p. 22 (equation between (24) and (25)).

    Simplification:

    .. math::

        θ = σ_p^2 + μ · (1 + σ_γ^2)

        → P = \\frac{1}{\\sqrt{2 π θ}} · \\exp\\left(- \\frac{(s - μ)^2}{2 θ}\\right)

        \\ln{P} = \\ln{\\frac{1}{\\sqrt{2 π θ}}} - \\frac{(s - μ)^2}{2 θ}

                = \\ln{1} - \\ln{\\sqrt{2 π θ}} - \\frac{(s - μ)^2}{2 θ}

                = - \\frac{\\ln{2 π θ}}{2} - \\frac{(s - μ)^2}{2 θ}

                = - \\frac{\\ln{2 π} + \\ln{θ}}{2} - \\frac{(s - μ)^2}{2 θ}

        - \\ln{P} = \\frac{\\ln{2 π} + \\ln{θ}}{2} + \\frac{(s - μ)^2}{2 θ}

    and since we can remove constants and factors in the minimization:

    .. math::

        - \\ln{P} = \\ln{θ} + \\frac{(s - μ)^2}{θ}


    Parameters
    ----------
    image: ndarray
        Pixel amplitudes from image (:math:`s`).
    prediction: ndarray
        Predicted pixel amplitudes from model (:math:`μ`).
    spe_width: ndarray
        Width of single p.e. peak (:math:`σ_γ`).
    pedestal: ndarray
        Width of pedestal (:math:`σ_p`).

    Returns
    -------
    float
    """
    theta = pedestal ** 2 + prediction * (1 + spe_width ** 2)

    neg_log_l = np.log(theta) + (image - prediction) ** 2 / theta

    return np.sum(neg_log_l)


def neg_log_likelihood_numeric(
    image, prediction, spe_width, pedestal, confidence=(0.001, 0.999)
):
    """
    Calculate likelihood of prediction given the measured signal,
    full numerical integration from [denaurois2009]_.

    Parameters
    ----------
    image: ndarray
        Pixel amplitudes from image (:math:`s`).
    prediction: ndarray
        Predicted pixel amplitudes from model (:math:`μ`).
    spe_width: ndarray
        Width of single p.e. peak (:math:`σ_γ`).
    pedestal: ndarray
        Width of pedestal (:math:`σ_p`).
    confidence: tuple(float, float), 0 < x < 1
        Confidence interval of poisson integration.

    Returns
    -------
    float
    """

    epsilon = np.finfo(np.float).eps

    prediction = prediction + epsilon

    likelihood = epsilon

    ns = np.arange(*poisson(np.max(prediction)).ppf(confidence))

    ns = ns[ns >= 0]

    for n in ns:
        theta = pedestal ** 2 + n * spe_width ** 2
        _l = (
            prediction ** n
            * np.exp(-prediction)
            / theta
            * np.exp(-((image - n) ** 2) / (2 * theta))
        )
        likelihood += _l

    return -np.sum(np.log(likelihood))


def neg_log_likelihood(image, prediction, spe_width, pedestal, prediction_safety=20.0):
    """
    Safe implementation of the poissonian likelihood implementation,
    adaptively switches between the full solution and the gaussian
    approx depending on the prediction. Prediction safety parameter
    determines cross over point between the two solutions.

    Parameters
    ----------
    image: ndarray
        Pixel amplitudes from image (:math:`s`).
    prediction: ndarray
        Predicted pixel amplitudes from model (:math:`μ`).
    spe_width: ndarray
        Width of single p.e. peak (:math:`σ_γ`).
    pedestal: ndarray
        Width of pedestal (:math:`σ_p`).
    prediction_safety: float
        Decision point to choose between poissonian likelihood
        and gaussian approximation.

    Returns
    -------
    float
    """

    approx_mask = prediction > prediction_safety

    neg_log_l = 0
    if np.any(approx_mask):
        neg_log_l += neg_log_likelihood_approx(
            image[approx_mask], prediction[approx_mask], spe_width, pedestal
        )

    if not np.all(approx_mask):
        neg_log_l += neg_log_likelihood_numeric(
            image[~approx_mask], prediction[~approx_mask], spe_width, pedestal
        )

    return neg_log_l


def mean_poisson_likelihood_gaussian(prediction, spe_width, pedestal):
    """Calculation of the mean likelihood for a give expectation
    value of pixel intensity in the gaussian approximation.
    This is useful in the calculation of the goodness of fit.

    Parameters
    ----------
    prediction: ndarray
        Predicted pixel amplitudes from model
    spe_width: ndarray
        Width of single p.e. distribution
    pedestal: ndarray
        Width of pedestal

    Returns
    -------
    float
    """
    theta = pedestal ** 2 + prediction * (1 + spe_width ** 2)
    mean_log_likelihood = 1 + np.log(2 * np.pi) + np.log(theta)

    return np.sum(mean_log_likelihood)


def _integral_poisson_likelihood_full(image, prediction, spe_width, ped):
    """
    Wrapper function around likelihood calculation, used in numerical
    integration.
    """
    image = np.asarray(image)
    prediction = np.asarray(prediction)
    like = neg_log_likelihood(image, prediction, spe_width, ped)
    return like * np.exp(-0.5 * like)


def mean_poisson_likelihood_full(prediction, spe_width, ped):
    """
    Calculation of the mean  likelihood for a give expectation value
    of pixel intensity using the full numerical integration.
    This is useful in the calculation of the goodness of fit.
    This numerical integration is very slow and really doesn't
    make a large difference in the goodness of fit in most cases.

    Parameters
    ----------
    prediction: ndarray
        Predicted pixel amplitudes from model
    spe_width: ndarray
        Width of single p.e. distribution
    pedestal: ndarray
        Width of pedestal

    Returns
    -------
    float
    """

    if len(spe_width) == 1:
        spe_width = np.full_like(prediction, spe_width)

    if len(ped) == 1:
        ped = np.full_like(prediction, ped)

    mean_like = 0

    width = ped ** 2 + prediction * spe_width ** 2
    width = np.sqrt(width)

    for pred, w, spe, p in zip(prediction, width, spe_width, ped):
        lower_integration_bound = pred - 10 * w
        upper_integration_bound = pred + 10 * w

        integral, *_ = quad(
            _integral_poisson_likelihood_full,
            lower_integration_bound,
            upper_integration_bound,
            args=(pred, spe, p),
            epsrel=0.05,
        )

        mean_like += integral

    return mean_like


def chi_squared(image, prediction, pedestal, error_factor=2.9):
    """
    Simple chi-squared statistic from Le Bohec et al 2008

    Parameters
    ----------
    image: ndarray
        Pixel amplitudes from image (:math:`s`).
    prediction: ndarray
        Predicted pixel amplitudes from model (:math:`μ`).
    pedestal: ndarray
        Width of pedestal (:math:`σ_p`).
    error_factor: float
        Ad-hoc error factor

    Returns
    -------
    float
    """

    chi_square = (image - prediction) ** 2 / (pedestal + 0.5 * (image - prediction))
    chi_square *= 1.0 / error_factor

    return np.sum(chi_square)
