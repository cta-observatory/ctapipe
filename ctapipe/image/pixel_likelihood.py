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
from scipy.special import factorial

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
    prediction = np.asarray(prediction)
    spe_width = np.asarray(spe_width)
    ped = np.asarray(ped)

    sq = 1.0 / np.sqrt(2 * np.pi * (ped ** 2 + prediction * (1 + spe_width ** 2)))

    diff = (image - prediction) ** 2
    denom = 2 * (ped ** 2 + prediction * (1 + spe_width ** 2))
    expo = np.asarray(np.exp(-1 * diff / denom))

    # If we are outside of the range of datatype, fix to lower bound
    min_prob = np.finfo(expo.dtype).tiny
    expo[expo < min_prob] = min_prob

    return -2 * np.log(sq * expo)


from numba import jit
#from numpy.math import factorial

@jit(nopython=True) 
def poisson_likelihood_full(
    image, prediction, spe_width, ped, width_fac=3, dtype=np.float32
):
    """
    Calculate likelihood of prediction given the measured signal,
    full numerical integration from de Naurois et al 2009.
    The width factor included here defines  the range over
    which photo electron contributions are summed, and is
    defined as a multiple of the expected resolution of
    the highest amplitude pixel. For most applications
    the defult of 3 is sufficient.
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
    
    shape = image.shape
    image = image.ravel()
    prediction = prediction.ravel()
    spe_width = spe_width.ravel()
    ped = ped.ravel()
    
    like = np.zeros_like(image)
    max_pix = np.max(image) + 100
    log_factorial = np.zeros(int(max_pix)+1)
    
    for i in range(1, int(max_pix)):
        fi = float(i)
        if i == 1:
            log_factorial[i] = np.log(fi)
        else:
            log_factorial[i] = log_factorial[i-1] + np.log(fi)
            
    for pix in range(0, image.shape[0]):

        max_sum = image[pix] + 100.

        total_probability = 0
        for pe in range(0, int(max_sum)):

            pe_probability = (pe*np.log(prediction[pix]) + -1*prediction[pix]) -(log_factorial[pe])
            pe_probability -= np.log(np.sqrt( 2*np.pi * (np.power(ped[pix],2) +(pe*np.power(spe_width[pix],2)))))
            pe_probability += -1 *( np.power(image[pix]-pe,2) / (2*(np.power(ped[pix],2) + (pe*np.power(spe_width[pix],2)))))

            if pe>image[pix]  and pe_probability<-40:
                break

            if(np.isnan(pe_probability) is False and total_probability!=0.0):
                total_probability=np.log(np.exp(total_probability-pe_probability)+1.0) + pe_probability
            else:
                total_probability=pe_probability

            if pe>image[pix] and total_probability<-40:
                break
            
        like[pix] = -2*total_probability

    return like.reshape(shape)


def poisson_likelihood(
    image,
    prediction,
    spe_width,
    ped,
    pedestal_safety=1.5,
    width_fac=5,
    dtype=np.float32,
):
    """
    Safe implementation of the poissonian likelihood implementation,
    adaptively switches between the full solution and the gaussian
    approx depending on the signal. Pedestal safety parameter 
    determines cross over point between the two solutions,
    based on the expected p.e. resolution of the image pixels.
    Therefore the cross over point will change dependent on 
    the single p.e. resolution and pedestal levels.
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
        Decision point to choose between poissonian likelihood 
        and gaussian approximation (p.e. resolution)
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
    prediction = np.asarray(prediction, dtype=dtype)
    spe_width = np.asarray(spe_width, dtype=dtype)
    ped = np.asarray(ped, dtype=dtype)

    # Calculate photoelectron resolution

    width = ped * ped + image * spe_width * spe_width
    width = np.asarray(width)
    width[width < 0] = 0  # Set width to 0 for negative pixel amplitudes
    width = np.sqrt(width)

    like = np.zeros(image.shape)
    # If larger than safety value use gaussian approx
    poisson_pix = width <= pedestal_safety
    gaus_pix = width > pedestal_safety

    if np.any(poisson_pix):
        like[poisson_pix] = poisson_likelihood_full(
            image[poisson_pix],
            prediction[poisson_pix],
            spe_width,
            ped,
            width_fac,
            dtype,
        )
    if np.any(gaus_pix):
        like[gaus_pix] = poisson_likelihood_gaussian(
            image[gaus_pix], prediction[gaus_pix], spe_width, ped
        )

    return like


def mean_poisson_likelihood_gaussian(prediction, spe_width, ped):
    """
    Calculation of the mean  likelihood for a give expectation
    value of pixel intensity in the gaussian approximation.
    This is useful in the calculation of the goodness of fit.
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
    ped = np.asarray(ped)

    mean_like = 1 + np.log(2 * np.pi)
    mean_like += np.log(ped * ped + prediction * (1 + spe_width * spe_width))

    return mean_like


def _integral_poisson_likelihood_full(s, prediction, spe_width, ped):
    """
    Wrapper function around likelihood calculation, used in numerical
    integration.
    """
    like = poisson_likelihood_full(np.array([s]), np.array([prediction]), 
                                   np.array([spe_width]), np.array([ped]), 
                                   width_fac=100)
    return like * np.exp(like/-2.)


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
    shape = prediction.shape
    prediction = prediction.ravel()
    spe_width = np.asarray(spe_width).ravel()
    ped = np.asarray(ped).ravel()

    if len(spe_width.shape) == 0:
        spe_width = np.ones(prediction.shape) * spe_width
    ped = np.asarray(ped)
    if len(ped.shape) == 0:
        ped = np.ones(prediction.shape) * ped
    mean_like = np.zeros(prediction.shape)

    for p in range(len(prediction)):
        #print(prediction[p], spe_width[p], ped[p])
        imin =  prediction[p] - 100
        if imin<-20:
            imin=-20
        int_range = (imin, prediction[p] + 100 )
        mean_like[p] = quad(
            _integral_poisson_likelihood_full,
            int_range[0],
            int_range[1],
            args=(prediction[p], spe_width[p], ped[p]),
            epsrel=0.001,
        )[0]
    return mean_like.reshape(shape)


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
