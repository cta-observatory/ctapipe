# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: UTF-8 -*-
"""
Hillas-style moment-based shower image parametrization.
"""

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle
from astropy.units import Quantity
from ..io.containers import HillasParametersContainer


__all__ = [
    'hillas_parameters',
    'HillasParameterizationError',
]


def camera_to_shower_coordinates(x, y, cog_x, cog_y, psi):
    '''
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
    '''
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
    image = np.ma.filled(image, 0)
    msg = 'Image and pixel shape do not match'
    assert pix_x.shape == pix_y.shape == image.shape, msg

    size = np.sum(image)

    if size == 0.0:
        raise HillasParameterizationError(
            'size=0, cannot calculate HillasParameters'
        )

    # calculate the cog as the mean of the coordinates weighted with the image
    cog_x = np.average(pix_x, weights=image)
    cog_y = np.average(pix_y, weights=image)

    # polar coordinates of the cog
    cog_r = np.linalg.norm([cog_x, cog_y])
    cog_phi = np.arctan2(cog_y, cog_x)

    # do the PCA for the hillas parameters
    delta_x = pix_x - cog_x
    delta_y = pix_y - cog_y

    # The ddof=0 makes this comparable to the other methods,
    # but ddof=1 should be more correct, mostly affects small showers
    # on a percent level
    cov = np.cov(delta_x, delta_y, aweights=image, ddof=0)
    eig_vals, eig_vecs = np.linalg.eigh(cov)

    # width and length are eigen values of the PCA
    width, length = np.sqrt(eig_vals)

    # psi is the angle of the eigenvector to length to the x-axis
    psi = np.arctan(eig_vecs[1, 1] / eig_vecs[0, 1])

    # calculate higher order moments along shower axes
    longitudinal = delta_x * np.cos(psi) + delta_y * np.sin(psi)

    m3_long = np.average(longitudinal**3, weights=image)
    skewness_long = m3_long / length**3

    m4_long = np.average(longitudinal**4, weights=image)
    kurtosis_long = m4_long / length**4

    return HillasParametersContainer(
        x=u.Quantity(cog_x, unit),
        y=u.Quantity(cog_y, unit),
        r=u.Quantity(cog_r, unit),
        phi=Angle(cog_phi, unit=u.rad),
        intensity=size,
        length=u.Quantity(length, unit),
        width=u.Quantity(width, unit),
        psi=Angle(psi, unit=u.rad),
        skewness=skewness_long,
        kurtosis=kurtosis_long,
    )
