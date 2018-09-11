# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: UTF-8 -*-
"""
Image timing-based shower image parametrization.
"""

from collections import namedtuple
import numpy as np
from astropy.units import Quantity

__all__ = [
    'TimingParameters',
    'timing_parameters'
]

TimingParameters = namedtuple(
    "TimingParameters",
    "gradient, intercept"
)


class TimingParameterizationError(RuntimeError):
    pass


def rotate_translate(pixel_pos_x, pixel_pos_y, psi):
    """
    Function to perform rotation and translation of pixel lists

    Parameters
    ----------
    pixel_pos_x: ndarray
        Array of pixel x positions
    pixel_pos_y: ndarray
        Array of pixel x positions
    phi: float
        Rotation angle of pixels

    Returns
    -------
        ndarray,ndarray: Transformed pixel x and y coordinates

    """
    # Here we are derotating, so we use a minus sign
    psi = -psi
    pixel_pos_rot_x = pixel_pos_x * np.cos(psi) - pixel_pos_y * np.sin(psi)
    pixel_pos_rot_y = pixel_pos_x * np.sin(psi) + pixel_pos_y * np.cos(psi)
    return pixel_pos_rot_x, pixel_pos_rot_y


def timing_parameters(pix_x, pix_y, image, peak_time, rotation_angle):
    """
    Function to extract timing parameters from a cleaned image

    Parameters
    ----------
    pix_x : array_like
        Pixel x-coordinate
    pix_y : array_like
        Pixel y-coordinate
    image : array_like
        Pixel values corresponding
    peak_time : array_like
        Pixel times corresponding
    rotation_angle: float
        Rotation angle for the image major axis, namely psi (in radians)

    Returns
    -------
    timing_parameters: TimingParameters
    """

    unit = Quantity(pix_x).unit
    pix_x = Quantity(np.asanyarray(pix_x, dtype=np.float64)).value
    pix_y = Quantity(np.asanyarray(pix_y, dtype=np.float64)).value
    image = np.asanyarray(image, dtype=np.float64)
    peak_time = np.asanyarray(peak_time, dtype=np.float64)

    # select only the pixels in the cleaned image that are greater than zero.
    # This is to allow to use a dilated mask (better for the time gradient calculation),
    # which eventually can consider also some pixel that have zero signal, which we need to exclude.

    mask = np.ma.masked_where(image > 0, image).mask
    pix_x = pix_x[mask]
    pix_y = pix_y[mask]
    image = image[mask]
    peak_time = peak_time[mask]

    assert pix_x.shape == image.shape
    assert pix_y.shape == image.shape
    assert peak_time.shape == image.shape

    # since the values of peak_pos are integers, sometimes the mask is constant for all the selected pixels
    # We can ask for a mask to have at least 3 different values for the peak_pos in the selected pixels

    if np.unique(peak_time).size > 2:
        pix_x_rot, pix_y_rot = rotate_translate(pix_x, pix_y, rotation_angle)
        gradient, intercept = np.polyfit(x=pix_x_rot, y=peak_time, deg=1, w=np.sqrt(image))
    else:
        gradient = 0.
        intercept = 0.

    return TimingParameters(gradient=gradient * (peak_time.unit / unit),
                            intercept=intercept * unit)