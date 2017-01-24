# Class for calculation of likelihood of a pixel expectation, given the pixel amplitude
# and typically the level of noise in the pixel
import numpy as np
import math
from scipy.misc import factorial


class PixelLikelihoodError(RuntimeError):
    pass


def poisson_likelihood_gaussian(image, prediction, spe_width, ped):
    """
    Calculate likelihood of prediction given the measured signal, gaussian approx from
    de Naurois et al 2009
    Parameters
    ----------
    image: ndarray
        Pixel amplitudes from image
    prediction: ndarray
        Predicted pixel amplitudes from model
    spe_width: ndarray
        width of single p.e. distributio
    ped: ndarray
        width of pedestal
    Returns
    -------
    ndarray: likelihood for each pixel
    """
    image = np.asarray(image)
    prediction= np.asarray(prediction)
    spe_width = np.asarray(spe_width)
    ped = np.asarray(ped)

    sq = 1. / np.sqrt(2 * math.pi * (np.power(ped, 2)
                                     + prediction * (1 + np.power(spe_width, 2))))

    diff = np.power(image - prediction, 2.)
    denom = 2 * (np.power(ped, 2) + prediction * (1 + np.power(spe_width, 2)))
    expo = np.exp(-1 * diff / denom)
    sm = expo < 1e-300
    expo[sm] = 1e-300

    return -2 * np.log(sq * expo)


def poisson_likelihood_full(image, prediction, spe_width, ped):
    """
    Calculate likelihood of prediction given the measured signal, full numerical integration from
    de Naurois et al 2009
    Parameters
    ----------
    image: ndarray
        Pixel amplitudes from image
    prediction: ndarray
        Predicted pixel amplitudes from model
    spe_width: ndarray
        width of single p.e. distribution
    ped: ndarray
        width of pedestal
    Returns
    -------
    ndarray: likelihood for each pixel
    """

    image = np.asarray(image)
    prediction= np.asarray(prediction)
    spe_width = np.asarray(spe_width)
    ped = np.asarray(ped)

    if image.shape is not prediction.shape:
        PixelLikelihoodError("Image and prediction arrays have different dimensions",
                             "Image shape: ",image.shape, "Prediction shape: ", prediction.shape)

    max_val = np.max(image)
    pe_summed = np.arange(max_val*10) # Need to decide how range is determined
    pe_factorial = factorial(pe_summed)

    first_term = np.power(prediction, pe_summed[:,np.newaxis]) * np.exp(-1*prediction)
    first_term /= pe_factorial[:,np.newaxis] * \
                  np.sqrt(math.pi*2 * (ped*ped + pe_summed[:,np.newaxis] * spe_width*spe_width))

    second_term = (image-pe_summed[:,np.newaxis])*(image-pe_summed[:,np.newaxis])
    second_term_denom = 2*(ped*ped + spe_width*spe_width*pe_summed[:,np.newaxis])

    second_term = second_term/second_term_denom
    second_term = np.exp(-1 * second_term)
    like = first_term * second_term
    return -2 * np.log(np.sum(like, axis=0))


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
    prediction= np.asarray(prediction)
    ped = np.asarray(ped)

    if image.shape is not prediction.shape:
        PixelLikelihoodError("Image and prediction arrays have different dimensions",
                             "Image shape: ",image.shape, "Prediction shape: ", prediction.shape)

    chi_square = (image-prediction)*(image-prediction)
    chi_square /= ped + 0.5*(image - prediction)
    chi_square *= 1./error_factor

    return chi_square