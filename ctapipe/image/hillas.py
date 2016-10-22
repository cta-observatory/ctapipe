# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Hillas shower parametrization.

TODO:
-----

* remove alpha calculation (which is only about (0,0), and make a get
  alpha function that does it from an arbitrary point given a
  pre-computed list of parameters
* At present skewness and kurtosis are calculated by hand.
  Is there a scipy way of doing weighted statistics?

CHANGE LOG:
-----------

* Higher order moments implemented in version 2.

* Fixed disagreement in psi between the two versions.

* Defined units for the returned parameters.
  Version 1 and version 2 are now compatible in units.

"""

import numpy as np
from astropy.units import Quantity
import astropy.units as u
from ctapipe.core import Container

__all__ = [
    'hillas_parameters',
    'HillasParameterizationError',
]

"""
hillas_parameters: Returns shower parameters Container upto 4th order

See also
--------
HillasParameterizationError
"""


class HillasParameterizationError(RuntimeError):
    pass


def hillas_parameters_1(pix_x, pix_y, image):
    """Compute Hillas parameters for a given shower image.

    Reference: Appendix of the Whipple Crab paper Weekes et al. (1998)
    http://adsabs.harvard.edu/abs/1989ApJ...342..379W
    (corrected for some obvious typos)

    Parameters
    ----------
    pix_x : array_like
        Pixel x-coordinate
    pix_y : array_like
        Pixel y-coordinate
    image : array_like
        Pixel values corresponding

    Returns
    -------
    hillas_parameters : `HillasContainer`
    """
    unit = Quantity(pix_x).unit

    pix_x = Quantity(np.asanyarray(pix_x, dtype=np.float64)).value
    pix_y = Quantity(np.asanyarray(pix_y, dtype=np.float64)).value
    image = np.asanyarray(image, dtype=np.float64)
    assert pix_x.shape == image.shape
    assert pix_y.shape == image.shape

    # Compute image moments
    size = np.sum(image)

    # Sanity check1:
    if size == 0:
        raise HillasParameterizationError(("Empty pixels! Cannot"
                                           " calculate image"
                                           " parameters. Exiting..."))

    mean_x = np.sum(image * pix_x) / size
    mean_y = np.sum(image * pix_y) / size

    # Compute major axis line representation y = a * x + b and correlations
    S_xx = np.sum(image * (pix_x - mean_x) ** 2) / size
    S_yy = np.sum(image * (pix_y - mean_y) ** 2) / size
    S_xy = np.sum(image * (pix_x - mean_x) * (pix_y - mean_y)) / size
    S_xxx = np.sum(image * (pix_x - mean_x) ** 3) / size
    S_yyy = np.sum(image * (pix_y - mean_y) ** 3) / size
    S_xyy = np.sum(image * (pix_x - mean_x) * (pix_y - mean_y) ** 2) / size
    S_xxy = np.sum(image * (pix_y - mean_y) * (pix_x - mean_x) ** 2) / size

    # Sanity check2:

    # If S_xy=0 (which should happen not very often, because Size>0)
    # we cannot calculate Length and Width.  In reallity it is almost
    # impossible to have a distribution of cerenkov photons in the
    # used pixels which is exactly symmetric along one of the axis
    if S_xy == 0:
        raise HillasParameterizationError(("X and Y uncorrelated. Cannot "
                                           "calculate length & width"))

    d0 = S_yy - S_xx
    d1 = 2 * S_xy
    # temp = d * d + 4 * S_xy * S_xy
    d2 = d0 + np.sqrt(d0 * d0 + d1 * d1)
    a = d2 / d1
    # Angle between ellipse major ax. and x-axis of camera. Will be used for
    # disp
    psi = ((np.pi / 2.0) + np.arctan(a))  # note: in radians
    b = mean_y - a * mean_x
    # Sin & Cos Will be used for calculating higher order image parameters
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)

    # Compute Hillas parameters
    width_2 = (S_yy + a * a * S_xx - 2 * a * S_xy) / \
        (1 + a * a)  # width squared
    length_2 = (S_xx + a * a * S_yy + 2 * a * S_xy) / \
        (1 + a * a)  # length squared

    # Sanity check3:
    width = 0. if width_2 < 0. else np.sqrt(width_2)
    length = 0. if length_2 < 0. else np.sqrt(length_2)

    miss = np.abs(b / np.sqrt(1 + a * a))
    r = np.sqrt(mean_x * mean_x + mean_y * mean_y)
    phi = np.arctan2(mean_y, mean_x)

    # Higher order moments
    sk = cos_psi * (pix_x - mean_x) + sin_psi * (pix_y - mean_y)

    skewness = ((np.sum(image * np.power(sk, 3)) / size) /
                ((np.sum(image * np.power(sk, 2)) / size) ** (3. / 2)))

    kurtosis = ((np.sum(image * np.power(sk, 4)) / size) /
                ((np.sum(image * np.power(sk, 2)) / size) ** 2))

    asym3 = (np.power(cos_psi, 3) * S_xxx
             + 3.0 * np.power(cos_psi, 2) * sin_psi * S_xxy
             + 3.0 * cos_psi * np.power(sin_psi, 2) * S_xyy
             + np.power(sin_psi, 3) * S_yyy)
    asym = - np.power(-asym3, 1. / 3) if (asym3 < 0.) \
        else np.power(asym3, 1. / 3)

    assert np.sign(skewness) == np.sign(asym)

    # another definition of assymetry asym = (mean_x -
    # pix_x[np.argmax(image)]) * cos_delta + (mean_y -
    # pix_y[np.argmax(image)]) * sin_delta

    HillasContainer = Container("HillasParams")
    HillasContainer.add_item("size", size)
    HillasContainer.add_item("cen_x", mean_x * unit)
    HillasContainer.add_item("cen_y", mean_y * unit)
    HillasContainer.add_item("length", length * unit)
    HillasContainer.add_item("width", width * unit)
    HillasContainer.add_item("dist", r * unit)
    HillasContainer.add_item("phi", (phi * u.rad).to(u.deg))
    HillasContainer.add_item("psi", (psi * u.rad).to(u.deg))
    HillasContainer.add_item("miss", miss * unit)
    HillasContainer.add_item("Skewness", skewness)
    HillasContainer.add_item("Kurtosis", kurtosis)
    HillasContainer.add_item("Asymmetry", asym)
    return HillasContainer


def hillas_parameters_2(pix_x, pix_y, image):
    """Compute Hillas parameters for a given shower image.

    Alternate implementation of `hillas_parameters` ...
    in the end we'll just keep one, but we're using Hilllas parameter
    computation as an example for performance checks.

    Parameters
    ----------
    pix_x : array_like
        Pixel x-coordinate
    pix_y : array_like
        Pixel y-coordinate
    image : array_like
        Pixel values corresponding

    Returns
    -------
    hillas_parameters : `HillasContainer`
    """

    unit = Quantity(pix_x).unit

    pix_x = Quantity(np.asanyarray(pix_x, dtype=np.float64)).value
    pix_y = Quantity(np.asanyarray(pix_y, dtype=np.float64)).value
    image = np.asanyarray(image)

    assert pix_x.shape == image.shape
    assert pix_y.shape == image.shape

    # Compute image moments (done in a bit faster way, but putting all
    # into one 2D array, where each row will be summed to calculate a
    # moment) However, this doesn't avoid a temporary created for the
    # 2D array

    size = image.sum()
    # Sanity check1:
    if size == 0:
        raise HillasParameterizationError(("Empty pixels! Cannot"
                                           " calculate image"
                                           " parameters. Exiting..."))
    momdata = np.row_stack([pix_x,
                            pix_y,
                            pix_x * pix_x,
                            pix_y * pix_y,
                            pix_x * pix_y,
                            pix_x * pix_x * pix_x,
                            pix_y * pix_y * pix_y,
                            pix_x * pix_y * pix_y,
                            pix_x * pix_x * pix_y]) * image

    moms = momdata.sum(axis=1) / size

    # calculate variances

    vx2 = moms[2] - moms[0] ** 2
    vy2 = moms[3] - moms[1] ** 2
    vxy = moms[4] - moms[0] * moms[1]
    vx3 = moms[5] - 3 * moms[2] * moms[0] + 2 * moms[0] * moms[0] * moms[0]
    vy3 = moms[6] - 3 * moms[3] * moms[1] + 2 * moms[1] * moms[1] * moms[1]
    vxy2 = moms[7] - moms[0] * moms[3] - 2 * moms[4] * moms[1] + 2 * moms[0] * moms[1] * moms[1]
    vx2y = moms[8] - moms[1] * moms[2] - 2 * moms[4] * moms[0] + 2 * moms[0] * moms[0] * moms[1]

    # Sanity check2:

    # If vxy=0 (which should happen not very often, because size>0)
    # we cannot calculate length and width.  In reallity it is almost
    # impossible to have a distribution of cerenkov photons in the
    # used pixels which is exactly symmetric along one of the axis
    if vxy == 0:
        raise HillasParameterizationError(("X and Y uncorrelated. Cannot "
                                           "calculate length & width"))

    # common factors:

    dd = vy2 - vx2
    zz = np.sqrt(dd ** 2 + 4.0 * vxy ** 2)

    # miss

    uu = 1.0 + dd / zz
    vv = 2.0 - uu
    miss = np.sqrt((uu * moms[0] ** 2 + vv * moms[1] ** 2) / 2.0
                   - moms[0] * moms[1] * 2.0 * vxy / zz)

    # shower shape parameters

    width = np.sqrt((vx2 + vy2 - zz) / 2.0)
    length = np.sqrt((vx2 + vy2 + zz) / 2.0)
    rr = np.hypot(moms[0], moms[1])

    # rotation angle of ellipse relative to centroid

    tanpsi_numer = (dd + zz) * moms[1] + 2.0 * vxy * moms[0]
    tanpsi_denom = (2 * vxy * moms[1]) - (dd - zz) * moms[0]
    psi = ((np.pi / 2.0) + np.arctan2(tanpsi_numer, tanpsi_denom))  # note: in radians
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)

    # polar coordinates of centroid

    phi = np.arctan2(moms[1], moms[0])

    # Higher order moments
    sk = cos_psi * (pix_x - moms[0]) + sin_psi * (pix_y - moms[1])

    highmom = np.row_stack([sk * sk,
                            sk * sk * sk,
                            sk * sk * sk * sk]) * image

    hmoms = highmom.sum(axis=1) / size

    skewness = hmoms[1] / hmoms[0] ** (3.0 / 2)
    kurtosis = hmoms[2] / hmoms[0] ** 2

    asym3 = (np.power(cos_psi, 3) * vx3
             + 3.0 * np.power(cos_psi, 2) * sin_psi * vx2y
             + 3.0 * cos_psi * np.power(sin_psi, 2) * vxy2
             + np.power(sin_psi, 3) * vy3)
    asym = - np.power(-asym3, 1. / 3) if (asym3 < 0.) \
        else np.power(asym3, 1. / 3)

    assert np.sign(skewness) == np.sign(asym)

    HillasContainer = Container("HillasParams")
    HillasContainer.add_item("size", size)
    HillasContainer.add_item("cen_x", moms[0] * unit)
    HillasContainer.add_item("cen_y", moms[1] * unit)
    HillasContainer.add_item("length", length * unit)
    HillasContainer.add_item("width", width * unit)
    HillasContainer.add_item("dist", rr * unit)
    HillasContainer.add_item("phi", (phi * u.rad).to(u.deg))
    HillasContainer.add_item("psi", (psi * u.rad).to(u.deg))
    HillasContainer.add_item("miss", miss * unit)
    HillasContainer.add_item("Skewness", skewness)
    HillasContainer.add_item("Kurtosis", kurtosis)
    HillasContainer.add_item("Asymmetry", asym)
    return HillasContainer

# use version 2 by default.
hillas_parameters = hillas_parameters_2
