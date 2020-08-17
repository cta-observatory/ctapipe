"""
Class for calculation of likelihood of a pixel expectation, given the pixel amplitude,
the level of noise in the pixel and the photoelectron resolution. This calculation is
taken from:
de Naurois & Rolland, Astroparticle Physics, Volume 32, Issue 5, p. 231-252 (2009)
https://arxiv.org/abs/0907.2610

The likelihood is essentially a poissonian convolved with a gaussian, at low signal
a full possonian approach must be adopted, which requires the sum of contibutions
over a number of potential contributing photoelectrons (which is slow and can fail
at high signals due to the factorial which mst be calculated). At high signal this
simplifies to a gaussian approximation.

The full and gaussian approximations are implemented, in addition to a general purpose
implementation, which tries to intellegently switch 
between the two. Speed tests are below:

poisson_likelihood_gaussian(image, prediction, spe, ped)
29.8 µs per loop

poisson_likelihood_full(image, prediction, spe, ped)
93.4 µs per loop

poisson_likelihood(image, prediction, spe, ped)
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
    "poisson_likelihood_gaussian",
    "poisson_likelihood_full",
    "poisson_likelihood",
    "mean_poisson_likelihood_gaussian",
    "mean_poisson_likelihood_full",
    "PixelLikelihoodError",
    "chi_squared",
]


class PixelLikelihoodError(RuntimeError):
    pass


def poisson_likelihood_gaussian(image, prediction, spe_width, pedestal):
    """Calculate negative log likelihood for every pixel.

    Gaussian approximation from [denaurois2009]_, p. 22 (between (24) and (25)).

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
    ndarray: Negative log-likelihood for each pixel.
    """
    theta = pedestal ** 2 + prediction * (1 + spe_width ** 2)

    neg_log_l = np.log(theta) + (image - prediction) ** 2 / theta

    return neg_log_l


def neg_log_likelihood_numeric(image, prediction, spe_width, pedestal, confidence=(0.001, 0.999)):
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

    ns = np.arange(
        *poisson(np.max(prediction)).ppf(confidence),
    )

    ns = ns[ns >= 0]

    for n in ns:
        theta = pedestal ** 2 + n * spe_width ** 2
        _l = prediction ** n * np.exp(-prediction) / theta * np.exp(-(image - n) ** 2 / (2 * theta))
        likelihood += _l

    return - np.sum(np.log(likelihood))


def neg_log_likelihood(
    image,
    prediction,
    spe_width,
    pedestal,
    prediction_safety=20.0,
):
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
    neg_log_l += neg_log_likelihood_approx(
        image[approx_mask],
        prediction[approx_mask],
        spe_width,
        pedestal,
    )

    neg_log_l += neg_log_likelihood_numeric(
        image[~approx_mask],
        prediction[~approx_mask],
        spe_width,
        pedestal,
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


def _integral_poisson_likelihood_full(s, prediction, spe_width, ped):
    """
    Wrapper function around likelihood calculation, used in numerical
    integration.
    """
    like = poisson_likelihood(s, prediction, spe_width, ped)
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
        width of single p.e. distribution
    ped: ndarray
        width of pedestal

    Returns
    -------
    ndarray: mean likelihood for give pixel expectation
    """
    prediction = np.asarray(prediction)
    spe_width = np.asarray(spe_width)

    if len(spe_width.shape) == 0:
        spe_width = np.ones(prediction.shape) * spe_width
    ped = np.asarray(ped)
    if len(ped.shape) == 0:
        ped = np.ones(prediction.shape) * ped
    mean_like = np.zeros(prediction.shape)
    width = ped * ped + prediction * spe_width * spe_width
    width = np.sqrt(width)

    for p in range(len(prediction)):
        int_range = (prediction[p] - 10 * width[p], prediction[p] + 10 * width[p])
        mean_like[p] = quad(
            _integral_poisson_likelihood_full,
            int_range[0],
            int_range[1],
            args=(prediction[p], spe_width[p], ped[p]),
            epsrel=0.05,
        )[0]
    return mean_like


def chi_squared(image, prediction, ped, error_factor=2.9):
    """
    Simple chi-squared statistic from Le Bohec et al 2008

    Parameters
    ----------
    image: ndarray
        Pixel amplitudes from image
    prediction: ndarray
        Predicted pixel amplitudes from model
    ped: ndarray
        width of pedestal
    error_factor: float
        ad hoc error factor

    Returns
    -------
    ndarray: likelihood for each pixel
    """

    image = np.asarray(image)
    prediction = np.asarray(prediction)
    ped = np.asarray(ped)

    if image.shape is not prediction.shape:
        PixelLikelihoodError(
            "Image and prediction arrays have different dimensions Image "
            "shape: {} Prediction shape: {}".format(image.shape, prediction.shape)
        )

    chi_square = (image - prediction) * (image - prediction)
    chi_square /= ped + 0.5 * (image - prediction)
    chi_square *= 1.0 / error_factor

    return chi_square
