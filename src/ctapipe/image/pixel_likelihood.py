"""
Class for calculation of likelihood of a pixel expectation, given the pixel amplitude,
the level of noise in the pixel and the photoelectron resolution.
This calculation is taken from :cite:p:`denaurois2009`.

The likelihood is essentially a poissonian convolved with a gaussian, at low signal
a full possonian approach must be adopted, which requires the sum of contributions
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
from scipy.special import factorial
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

EPSILON = 5.0e-324


class PixelLikelihoodError(RuntimeError):
    pass


def neg_log_likelihood_approx(image, prediction, spe_width, pedestal):
    """Calculate negative log likelihood for telescope.

    Gaussian approximation from :cite:p:`denaurois2009`, p. 22 (equation between (24) and (25)).

    Simplification:

    .. math::

        θ = σ_p^2 + μ · (1 + σ_γ^2)

        → P = \\frac{1}{\\sqrt{2 π θ}} · \\exp\\left(- \\frac{(s - μ)^2}{2 θ}\\right)

        \\ln{P} = \\ln{\\frac{1}{\\sqrt{2 π θ}}} - \\frac{(s - μ)^2}{2 θ}

                = \\ln{1} - \\ln{\\sqrt{2 π θ}} - \\frac{(s - μ)^2}{2 θ}

                = - \\frac{\\ln{2 π θ}}{2} - \\frac{(s - μ)^2}{2 θ}

                = - \\frac{\\ln{2 π} + \\ln{θ}}{2} - \\frac{(s - μ)^2}{2 θ}

        - \\ln{P} = \\frac{\\ln{2 π} + \\ln{θ}}{2} + \\frac{(s - μ)^2}{2 θ}

    We keep the constants in this because the actual value of the likelihood
    can be used to calculate a goodness-of-fit value


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
    ndarray
    """

    theta = 2 * (pedestal**2 + prediction * (1 + spe_width**2))
    neg_log_l = np.log(np.pi * theta) / 2.0 + (image - prediction) ** 2 / theta

    return neg_log_l


def neg_log_likelihood_numeric(
    image, prediction, spe_width, pedestal, confidence=0.999
):
    """
    Calculate likelihood of prediction given the measured signal,
    full numerical integration from :cite:p:`denaurois2009`.

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
    confidence: float, 0 < x < 1
        Upper end of Poisson confidence interval of maximum prediction.
        Determines upper end of poisson integration.

    Returns
    -------
    ndarray
    """

    epsilon = np.finfo(np.float64).eps

    prediction = prediction + epsilon

    likelihood = np.full_like(prediction, epsilon, dtype=np.float64)

    n_signal = np.arange(poisson(np.max(prediction)).ppf(confidence) + 1)

    n_signal = n_signal[n_signal >= 0]

    for n in n_signal:
        theta = pedestal**2 + n * spe_width**2
        _l = (
            prediction**n
            * np.exp(-prediction)
            / np.sqrt(2 * np.pi * theta)
            / factorial(n)
            * np.exp(-((image - n) ** 2) / (2 * theta))
        )
        likelihood += _l

    return -np.log(likelihood)


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
    ndarray
    """

    approx_mask = prediction > prediction_safety

    neg_log_l = np.zeros_like(image, dtype=np.float64)
    if np.any(approx_mask):
        neg_log_l[approx_mask] += neg_log_likelihood_approx(
            image[approx_mask],
            prediction[approx_mask],
            spe_width[approx_mask],
            pedestal[approx_mask],
        )

    if not np.all(approx_mask):
        neg_log_l[~approx_mask] += neg_log_likelihood_numeric(
            image[~approx_mask],
            prediction[~approx_mask],
            spe_width[~approx_mask],
            pedestal[~approx_mask],
        )

    return neg_log_l


def mean_poisson_likelihood_gaussian(prediction, spe_width, pedestal):
    """Calculation of the mean of twice the negative log likelihood for a give expectation
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
    ndarray
    """
    theta = pedestal**2 + prediction * (1 + spe_width**2)
    mean_log_likelihood = 1 + np.log(2 * np.pi) + np.log(theta + EPSILON)

    return mean_log_likelihood


def _integral_poisson_likelihood_full(image, prediction, spe_width, ped):
    """
    Wrapper function around likelihood calculation, used in numerical
    integration.
    """
    image = np.asarray(image)
    prediction = np.asarray(prediction)
    like = neg_log_likelihood(image, prediction, spe_width, ped)
    return 2 * like * np.exp(-like)


def mean_poisson_likelihood_full(prediction, spe_width, ped):
    """
    Calculation of the mean of twice the negative log likelihood for a give expectation value
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
    ndarray
    """

    if len(spe_width) == 1:
        spe_width = np.full_like(prediction, spe_width, dtype=np.float64)

    if len(ped) == 1:
        ped = np.full_like(prediction, ped, dtype=np.float64)

    mean_like = np.zeros_like(prediction, dtype=np.float64)

    width = ped**2 + prediction * spe_width**2
    width = np.sqrt(width)

    for i, (pred, w, spe, p) in enumerate(zip(prediction, width, spe_width, ped)):
        lower_integration_bound = pred - 10 * w
        upper_integration_bound = pred + 10 * w

        integral, *_ = quad(
            _integral_poisson_likelihood_full,
            lower_integration_bound,
            upper_integration_bound,
            args=(pred, spe, p),
            epsrel=0.05,
        )

        mean_like[i] = integral

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
    ndarray
    """

    chi_square = (image - prediction) ** 2 / (pedestal + 0.5 * (image - prediction))
    chi_square *= 1.0 / error_factor

    return chi_square
