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

    min_prob = np.finfo(expo.dtype).tiny # If we are outside of the range of datatype, fix to lower bound
    expo[expo<min_prob] = min_prob

    return -2 * np.log(sq * expo)


def poisson_likelihood_full(image, prediction, spe_width, ped, width_fac=3, dtype=np.float32):
    """
    Calculate likelihood of prediction given the measured signal, full numerical integration from
    de Naurois et al 2009. The width factor included here defines the range over which photo electron contributions
    are summed, and is defined as a multiple of the expected resolution of the highest amplitude pixel. For
    most applications the defult of 3 is sufficient.

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
    width_fac: float
        Factor to determine range of summation on integral
    dtype: datatype
        Data type of output array
    Returns
    -------
    ndarray: likelihood for each pixel
    """

    image = np.asarray(image, dtype=dtype)
    prediction= np.asarray(prediction, dtype=dtype)
    spe_width = np.asarray(spe_width, dtype=dtype)
    ped = np.asarray(ped, dtype=dtype)

    if image.shape[0] != prediction.shape[0]:
        raise PixelLikelihoodError("Image and prediction arrays have different dimensions",
                                   "Image shape: ",image.shape[0], "Prediction shape: ", prediction.shape[0])

    max_val = np.max(image)

    max_sum = max_val + width_fac*np.sqrt(ped*ped + max_val*spe_width*spe_width)
    if max_sum<5:
        max_sum = 5

    pe_summed = np.arange(max_sum) # Need to decide how range is determined
    pe_factorial = factorial(pe_summed)

    first_term = np.power(prediction, pe_summed[:,np.newaxis]) * np.exp(-1*prediction)
    first_term /= pe_factorial[:,np.newaxis] * \
                  np.sqrt(math.pi*2 * (ped*ped + pe_summed[:,np.newaxis] * spe_width*spe_width))

    # Throw error if we get NaN in likelihood
    if np.any(np.isnan(first_term)):
        raise PixelLikelihoodError("Likelihood returning NaN, likely due to extremely high signal"
                                   " deviation. Switch to poisson_likelihood_safe implementation or"
                                   " increase floating point precision e.g. dtype=float64")

    # Should not have any porblems here with NaN that have not bee seens
    second_term = (image-pe_summed[:,np.newaxis])*(image-pe_summed[:,np.newaxis])
    second_term_denom = 2*(ped*ped + spe_width*spe_width*pe_summed[:,np.newaxis])

    second_term = second_term/second_term_denom
    second_term = np.exp(-1 * second_term)

    min_prob = np.finfo(second_term.dtype).tiny # If we are outside of the range of datatype, fix to lower bound
    second_term[second_term<min_prob] = min_prob

    like = first_term * second_term
    return -2 * np.log(np.sum(like, axis=0))


def poisson_likelihood(image, prediction, spe_width, ped, pedestal_safety=1.5, width_fac=3, dtype=np.float32):
    """
    Safe implementation of the poissonian likelihood implementation , adaptively switches between the full
    solution and the gaussian approx depending on the signal. Pedestal safety parameter determines cross
    over point between the two solutions, based on the expected p.e. resolution of the image pixels. Therefore
    the cross over point will change dependent on the single p.e. resolution and pedestal levels.

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
    pedestal_safety: float
        Decision point to choose between poissonian likelihood and gaussian approximation (p.e. resolution)
    width_fac: float
        Factor to determine range of summation on integral
    dtype: datatype
        Data type of output array
    Returns
    -------
    ndarray: pixel likelihoods
    """
    # Convert everything to arrays to begin
    image = np.asarray(image, dtype=dtype)
    prediction= np.asarray(prediction, dtype=dtype)
    spe_width = np.asarray(spe_width, dtype=dtype)
    ped = np.asarray(ped, dtype=dtype)

    # Calculate photoelectron resolution

    width = ped*ped + image*spe_width*spe_width
    width[width<0] = 0 # Set width to 0 for negative pixel amplitudes
    width = np.sqrt(width)

    like = np.zeros(image.shape)
    # If larger than safety value use gaussian approx
    poisson_pix = width<pedestal_safety
    gaus_pix = width>pedestal_safety

    like[poisson_pix] = poisson_likelihood_full(image[poisson_pix], prediction[poisson_pix],
                                                spe_width, ped, width_fac, dtype)

    like[gaus_pix] = poisson_likelihood_gaussian(image[gaus_pix], prediction[gaus_pix],
                                                 spe_width, ped)

    return like

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