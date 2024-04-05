# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: UTF-8 -*-
"""
Hillas-style moment-based shower image parametrization.
"""

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle

from ..containers import CameraHillasParametersContainer, HillasParametersContainer

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

    The recommended form is to pass only the sliced geometry and image
    for the pixels to be considered.

    Each method gives the same result, but vary in efficiency

    Parameters
    ----------
    geom: ctapipe.instrument.CameraGeometry
        Camera geometry, the cleaning mask should be applied to improve performance
    image : array_like
        Charge in each pixel, the cleaning mask should already be applied to
        improve performance.

    Returns
    -------
    HillasParametersContainer:
        container of hillas parametesr
    """
    unit = geom.pix_x.unit
    pix_x = geom.pix_x.to_value(unit)
    pix_y = geom.pix_y.to_value(unit)
    image = np.asanyarray(image, dtype=np.float64)

    if isinstance(image, np.ma.masked_array):
        image = np.ma.filled(image, 0)

    if not (pix_x.shape == pix_y.shape == image.shape):
        raise ValueError("Image and pixel shape do not match")

    size = np.sum(image)

    if size == 0.0:
        raise HillasParameterizationError("size=0, cannot calculate HillasParameters")

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

    # round eig_vals to get rid of nans when eig val is something like -8.47032947e-22
    near_zero = np.isclose(eig_vals, 0, atol=HILLAS_ATOL)
    eig_vals[near_zero] = 0

    # width and length are eigen values of the PCA
    width, length = np.sqrt(eig_vals)

    # psi is the angle of the eigenvector to length to the x-axis
    vx, vy = eig_vecs[0, 1], eig_vecs[1, 1]

    # avoid divide by 0 warnings
    # psi will be consistently defined in the range (-pi/2, pi/2)
    if length == 0:
        psi = skewness_long = kurtosis_long = np.nan
    else:
        if vx != 0:
            psi = np.arctan(vy / vx)
        else:
            psi = np.pi / 2

        # calculate higher order moments along shower axes
        longitudinal = delta_x * np.cos(psi) + delta_y * np.sin(psi)

        m3_long = np.average(longitudinal**3, weights=image)
        skewness_long = m3_long / length**3

        m4_long = np.average(longitudinal**4, weights=image)
        kurtosis_long = m4_long / length**4

    # Compute of the Hillas parameters uncertainties.
    # Implementation described in [hillas_uncertainties]_ This is an internal MAGIC document
    # not generally accessible.

    # intermediate variables
    cos_2psi = np.cos(2 * psi)
    a = (1 + cos_2psi) / 2
    b = (1 - cos_2psi) / 2
    c = np.sin(2 * psi)

    A = ((delta_x**2.0) - cov[0][0]) / size
    B = ((delta_y**2.0) - cov[1][1]) / size
    C = ((delta_x * delta_y) - cov[0][1]) / size

    # Hillas's uncertainties

    # avoid divide by 0 warnings
    if length == 0:
        length_uncertainty = np.nan
    else:
        length_uncertainty = np.sqrt(
            np.sum((((a * A) + (b * B) + (c * C)) ** 2.0) * image)
        ) / (2 * length)

    if width == 0:
        width_uncertainty = np.nan
    else:
        width_uncertainty = np.sqrt(
            np.sum((((b * A) + (a * B) + (-c * C)) ** 2.0) * image)
        ) / (2 * width)

    if unit.is_equivalent(u.m):
        return CameraHillasParametersContainer(
            x=u.Quantity(cog_x, unit),
            y=u.Quantity(cog_y, unit),
            r=u.Quantity(cog_r, unit),
            phi=Angle(cog_phi, unit=u.rad),
            intensity=size,
            length=u.Quantity(length, unit),
            length_uncertainty=u.Quantity(length_uncertainty, unit),
            width=u.Quantity(width, unit),
            width_uncertainty=u.Quantity(width_uncertainty, unit),
            psi=Angle(psi, unit=u.rad),
            skewness=skewness_long,
            kurtosis=kurtosis_long,
        )
    return HillasParametersContainer(
        fov_lon=u.Quantity(cog_x, unit),
        fov_lat=u.Quantity(cog_y, unit),
        r=u.Quantity(cog_r, unit),
        phi=Angle(cog_phi, unit=u.rad),
        intensity=size,
        length=u.Quantity(length, unit),
        length_uncertainty=u.Quantity(length_uncertainty, unit),
        width=u.Quantity(width, unit),
        width_uncertainty=u.Quantity(width_uncertainty, unit),
        psi=Angle(psi, unit=u.rad),
        skewness=skewness_long,
        kurtosis=kurtosis_long,
    )
