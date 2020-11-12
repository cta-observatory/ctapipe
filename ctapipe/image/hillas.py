# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: UTF-8 -*-
"""
Hillas-style moment-based shower image parametrization.
"""

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle
from astropy.units import Quantity
from numba import njit
from ..containers import HillasParametersContainer


HILLAS_ATOL = np.finfo(np.float64).eps


__all__ = ["hillas_parameters", "HillasParameterizationError"]


def camera_to_shower_coordinates(x, y, cog_x, cog_y, psi):
    """
    Return longitudinal and transverse coordinates for x and y
    for a given set of hillas parameters

    Parameters
    ----------
    x: u.Quantity[length]
        x coordinate in camera coordinates
    y: u.Quantity[length]
        y coordinate in camera coordinates
    cog_x: u.Quantity[length]
        x coordinate of center of gravity
    cog_y: u.Quantity[length]
        y coordinate of center of gravity
    psi: Angle
        orientation angle

    Returns
    -------
    longitudinal: astropy.units.Quantity
        longitudinal coordinates (along the shower axis)
    transverse: astropy.units.Quantity
        transverse coordinates (perpendicular to the shower axis)
    """
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)

    delta_x = x - cog_x
    delta_y = y - cog_y

    longi = delta_x * cos_psi + delta_y * sin_psi
    trans = delta_x * -sin_psi + delta_y * cos_psi

    return longi, trans


class HillasParameterizationError(RuntimeError):
    pass


def hillas_parameters(geom, image):
    """
    Compute Hillas parameters for a given shower image.

    Implementation uses a PCA analogous to the implementation in
    src/main/java/fact/features/HillasParameters.java
    from
    https://github.com/fact-project/fact-tools

    The image passed to this function can be in three forms:

    >>> from ctapipe.image.hillas import hillas_parameters
    >>> from ctapipe.image.tests.test_hillas import create_sample_image, compare_hillas
    >>> geom, image, clean_mask = create_sample_image(psi='0d')
    >>>
    >>> # Fastest
    >>> geom_selected = geom[clean_mask]
    >>> image_selected = image[clean_mask]
    >>> hillas_selected = hillas_parameters(geom_selected, image_selected)
    >>>
    >>> # Mid (1.45 times longer than fastest)
    >>> image_zeros = image.copy()
    >>> image_zeros[~clean_mask] = 0
    >>> hillas_zeros = hillas_parameters(geom, image_zeros)
    >>>
    >>> # Slowest (1.51 times longer than fastest)
    >>> image_masked = np.ma.masked_array(image, mask=~clean_mask)
    >>> hillas_masked = hillas_parameters(geom, image_masked)
    >>>
    >>> compare_hillas(hillas_selected, hillas_zeros)
    >>> compare_hillas(hillas_selected, hillas_masked)

    Each method gives the same result, but vary in efficiency

    Parameters
    ----------
    geom: ctapipe.instrument.CameraGeometry
        Camera geometry
    image : array_like
        Charge in each pixel

    Returns
    -------
    HillasParametersContainer:
        container of hillas parametesr
    """
    unit = geom.pix_x.unit
    pix_x = Quantity(np.asanyarray(geom.pix_x, dtype=np.float64)).value
    pix_y = Quantity(np.asanyarray(geom.pix_y, dtype=np.float64)).value
    image = np.asanyarray(image, dtype=np.float64)
    if np.ma.is_masked(image):
        raise TypeError(
            "np.ma.masked arrays are not supported by hillas_parameters(). "
            "Use `hillas_parameters(geom[mask], image[mask])` instead."
        )
    msg = "Image and pixel shape do not match"
    assert pix_x.shape == pix_y.shape == image.shape, msg

    # if size == 0.0:
    #    raise HillasParameterizationError("size=0, cannot calculate HillasParameters")

    (
        cog_x,
        cog_y,
        cog_r,
        cog_phi,
        size,
        length,
        length_uncertainty,
        width,
        width_uncertainty,
        psi_rad,
        skewness_long,
        kurtosis_long,
    ) = hillas_parameters_fast(pix_x, pix_y, image)

    if np.isnan(size):
        raise HillasParameterizationError(
            "intensity=0, cannot calculate HillasParameters"
        )

    return HillasParametersContainer(
        x=u.Quantity(cog_x, unit),
        y=u.Quantity(cog_y, unit),
        r=u.Quantity(cog_r, unit),
        phi=Angle(cog_phi, unit=u.rad),
        intensity=size,
        length=u.Quantity(length, unit),
        length_uncertainty=u.Quantity(length_uncertainty, unit),
        width=u.Quantity(width, unit),
        width_uncertainty=u.Quantity(width_uncertainty, unit),
        psi=Angle(psi_rad, unit=u.rad),
        skewness=skewness_long,
        kurtosis=kurtosis_long,
    )


@njit
def weighted_average_1d(values, weights):
    """
    since Numba doesn't support np.average(), this is a simplistic version
    without the various checks and broadcasting
    """
    scale = weights.sum()
    return np.multiply(values, weights).sum() / scale


@njit
def covariance_2d(delta_x, delta_y, weights_normed):
    """covariance assuming x and y are already weighted-mean subtracted
    and the weights are normalized"""
    w2_sum = (weights_normed ** 2).sum()
    if w2_sum == 1:
        return 0

    return np.sum(weights_normed * delta_x * delta_y) / (1.0 - w2_sum)


@njit
def covariance_matrix_2d(x, y, weights):
    """
    covariance matrix in 2d, assuming that x and y are already
    weighed-mean subtracted
    """
    # fmt: off
    w_sum =  weights.sum()
    if w_sum == 0.0:
        return np.array([[0,0],[0,0]], dtype=np.float64)

    w_normed = weights  / w_sum
    return np.array(
        [[covariance_2d(x, x, w_normed), covariance_2d(x, y, w_normed)],
         [covariance_2d(y, x, w_normed), covariance_2d(y, y, w_normed)]]
    )
    # fmt: on


@njit
def hillas_parameters_fast(pix_x, pix_y, image):
    """
    A helper function to Compute hillas paremeters rapidly using Numba. This
    function assumes pure unitless numpy arrays as input.
    """
    size = np.sum(image)
    if size == 0.0:
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    # calculate the cog as the mean of the coordinates weighted with the image
    cog_x = weighted_average_1d(pix_x, weights=image)
    cog_y = weighted_average_1d(pix_y, weights=image)

    # polar coordinates of the cog
    cog2d = np.asarray([cog_x, cog_y])
    cog_r = np.linalg.norm(cog2d)
    cog_phi = np.arctan2(cog_y, cog_x)

    # do the PCA for the hillas parameters
    delta_x = pix_x - cog_x
    delta_y = pix_y - cog_y

    # The ddof=0 makes this comparable to the other methods,
    # but ddof=1 should be more correct, mostly affects small showers
    # on a percent level
    # cov = np.cov(delta_x, delta_y, aweights=image, ddof=0)
    cov = covariance_matrix_2d(delta_x, delta_y, image)
    eig_vals, eig_vecs = np.linalg.eigh(cov)

    # round eig_vals to get rid of nans when eig val is something like -8.47032947e-22
    near_zero = np.abs(eig_vals) < HILLAS_ATOL
    eig_vals[near_zero] = 0

    # width and length are eigen values of the PCA
    width, length = np.sqrt(eig_vals)

    # psi is the angle of the eigenvector to length to the x-axis
    vx, vy = eig_vecs[0, 1], eig_vecs[1, 1]

    # avoid divide by 0 warnings
    if length == 0:
        psi_rad = skewness_long = kurtosis_long = np.nan
    else:
        if vx != 0:
            psi_rad = np.arctan(vy / vx)
        else:
            psi_rad = np.pi / 2

        # calculate higher order moments along shower axes
        longitudinal = delta_x * np.cos(psi_rad) + delta_y * np.sin(psi_rad)

        m3_long = weighted_average_1d(longitudinal ** 3, weights=image)
        skewness_long = m3_long / length ** 3

        m4_long = weighted_average_1d(longitudinal ** 4, weights=image)
        kurtosis_long = m4_long / length ** 4

    # Hillas's uncertainties
    length_uncertainty, width_uncertainty = compute_uncertainties(
        image, delta_x, delta_y, size, length, width, psi_rad, cov
    )

    return (
        cog_x,
        cog_y,
        cog_r,
        cog_phi,
        size,
        length,
        length_uncertainty,
        width,
        width_uncertainty,
        psi_rad,
        skewness_long,
        kurtosis_long,
    )


@njit
def compute_uncertainties(image, delta_x, delta_y, size, length, width, psi_rad, cov):
    """
    Compute of the Hillas parameters uncertainties. Implementation described in
    [hillas_uncertainties]_ This is an internal MAGIC document not generally
    accessible

    Parameters
    ----------
    image: np.ndarray
        image values
    delta_x: np.ndarray[float]
        x distance of pixels to cog
    delta_y: np.ndarray[float]
        y distance of pixels to cog
    size: float
        size Hillas parameter (total intensity)
    length: float
        length Hillas parameter
    width: float
        width Hillas parameter
    psi_rad: float
        psi hillas parameter in radians
    cov: np.ndarray[float]
        covariance matrix used to compute width, length
    """

    # intermediate variables
    cos_2psi = np.cos(2 * psi_rad)
    a = (1 + cos_2psi) / 2
    b = (1 - cos_2psi) / 2
    c = np.sin(2 * psi_rad)

    A = ((delta_x ** 2.0) - cov[0][0]) / size
    B = ((delta_y ** 2.0) - cov[1][1]) / size
    C = ((delta_x * delta_y) - cov[0][1]) / size

    length_uncertainty = np.nan
    width_uncertainty = np.nan

    # avoid divide by 0 warnings
    if length != 0:
        length_uncertainty = np.sqrt(
            np.sum(((((a * A) + (b * B) + (c * C))) ** 2.0) * image)
        ) / (2 * length)

    if width != 0:
        width_uncertainty = np.sqrt(
            np.sum(((((b * A) + (a * B) + (-c * C))) ** 2.0) * image)
        ) / (2 * width)

    return length_uncertainty, width_uncertainty
