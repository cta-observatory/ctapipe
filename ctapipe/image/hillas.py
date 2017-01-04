# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: UTF-8 -*-
"""Hillas shower parametrization.

TODO:
=====

* remove alpha calculation (which is only about (0,0)),
    and make a get alpha function that does it from an arbitrary point given a pre-computed list of parameters
    (has this alpha function been implemented?  V. easy, given centroid and psi, or miss and dist/r)
    (logically, should also remove "miss", which is relative to (0,0)
* add function(s) to calculate asymmetry (or asymmetries?) including sign calculation
    (for now, MP/Whipple asymmetry is positive for "tail" pointing away from (0,0))
    (I have a wonderful way to to this, but this margin is too small to contain it)
* psi of hillas_parameters_1 and 2 does not match. Mismatch in hillas_parameters_2 need to be resolved:
    psi of hillas_parameters_1 has inconsistent sign.  hillas_parameters_2/3/4 better, since +ve for tail pointing away
    Mismatch between hillas_parameters_1/2 fixed in previous version, shouldn't this TODO be removed?
* Implement higher order hillas parameters in hillas_parameters_2: Done (MP)

CHANGE LOG:
===========
* MP: Speed comparison for an image in a 960 pixel camera for image created with "mock",
    %timeit -n 10000
    hillas_parameters_1(pix_x, pix_y, image, higherMoments=True)                        best of 3:     598 µs per loop
    hillas_parameters_1(pix_x, pix_y, image, higherMoments=False)                       best of 3:     391 µs per loop

    hillas_parameters_2(pix_x, pix_y, image, higherMoments=True, reCalcPix=True)        best of 3:     210 µs per loop
    hillas_parameters_2(pix_x, pix_y, image, higherMoments=True, reCalcPix=False)       best of 3:     137 µs per loop
    hillas_parameters_2(pix_x, pix_y, image, higherMoments=False, reCalcPix=True)       best of 3:     125 µs per loop
    hillas_parameters_2(pix_x, pix_y, image, higherMoments=False, reCalcPix=False)      best of 3:      95.9 µs per loop

    hillas_parameters_3(pix_x, pix_y, image, higherMoments=True)                        best of 3:   1.55 ms per loop
    hillas_parameters_3(pix_x, pix_y, image, higherMoments=False)                       best of 3:   1.5 ms per loop

    hillas_parameters_4(pix_x, pix_y, image, higherMoments=True, reCalcPix=True)        best of 3:     130 µs per loop
    hillas_parameters_4(pix_x, pix_y, image, higherMoments=True, reCalcPix=False)       best of 3:     110 µs per loop
    hillas_parameters_4(pix_x, pix_y, image, higherMoments=False, reCalcPix=True)       best of 3:      70.4 µs per loop
    hillas_parameters_4(pix_x, pix_y, image, higherMoments=False, reCalcPix=False)      best of 3:      62.7 µs per loop

    (any way to compare compiled versions?)

* MP: all routines tested with +/- combinations and x/y swaps of "mini-camera"
        pix_x = array([1.1, 1.1, 1, 1, 1, 1, 1, 1])
        pix_y = array([0, 1, 2, 3, 4, 5, 6, 7])
        image = array([0., 10, 3, 5, 7, 6, 2, 4 ])
    TODO: add this as a test, with known results (change sign of psi where needed):
        (MomentParameters(size=37.0, cen_x=1.027027027027027, cen_y=3.4864864864864864,
        length=1.9951647036522218, width=0.028933760477638449, r=3.6346076177074274,
        phi=1.2843252459918457, psi=1.5876852050217887, miss=1.0857606054046072),
        HighOrderMomentParameters(Skewness=0.21338802620071526, Kurtosis=1.9087015993786907))

* MP: removed returning of asymmetry, just left skewness...
    either MP/Whipple or MAGIC asymmetry can be calculated then in function

* MP: added two Boolean variables to the call: higherMoments (default True) and ReCalcPix (default True)
    If higherMoments is False, then it doesn't calculate them (if True, calculated for those routines have them)
    If reCalcPix is False, then the multiples of pixel positions are calculated always (if True, only if not existing)

* MP: hillas_parameter_1 doesn't work for the case of horizontal image
    It raises an exception wrongly; no reason to raise exception for S_xy = 0

* MP: In hillas_parameter_1, it says delta/psi is angle with x-axis, but adds pi/2 (as does hillas_parameter_2).
    cos_delta and sin_delta are correct, though.
     => changed this in hillas_parameter_1, and in modified the display visualization (in visualization/mpl.py)... and "mock.py" in reco.

* MP: Added higher moments to hillas_parameters_2, and added a sub-function to hold static variables
    of pixel multiplications, if pixel positions don't change

* MP: Added Pythonified version of hillas_parameters_3, called hillal_parameters_4, also using
    a sub-function's attributes to hold "static variables" of pixel multiplications, if pixel positions don't change

* MP: Added the historic 1993 Whipple calculation (translated Fortran code via c), as hillas_parameters_3
    Note, MAGIC's "skewness" is the same as the Whipple/MP "asymmetry^3", why ???
    ... and also, Whipple/MP "asymmetry" * "length" = MAGIC "asymmetry"
    ... so, MAGIC "asymmetry" = MAGIC "skewness"^(1/3) * "length"
    I don't know what MAGIC's "asymmetry" is supposed to be.

* MP: MAGIC (hillas_parameters_1) seems to be using the Whipple 1989 Weekes et al. formula calculation!
  The Whipple 1993 Reynolds et al formula calculation is better (using non central or raw moments, not central moments).

* Higher order moments need not be explicitly defined.
    Only mean and size parameters are enough to define correlations of all order.
    Implemented in the same way as in MAGIC.

* Third and fourth order correlations implemented in hillas_1.

* Sanity checks for size, x_y correlation
  (HillasParameterizationError), length, width.

* Parameter psi introduced in hillas_parameters_1: angle between
  ellipse major axis and camera x-axis.

* Correction in implementation of miss parameter of hillas_parameters_1.
    Previous implementation missed a square root factor.

* Correction in implementation of length, width in
  hillas_parameters_2. Previous version missed a 2 in the denominator.

* Implementation of Higher Order Moment Parameters in
  hillas_parameters_1: Skewness, Kurtosis.

* Implementation of Asymmetry. Alternative definition of asym using
  higher order correlations mentioned in comments below asym.

"""

from collections import namedtuple
import numpy as np
from astropy.units import Quantity
import astropy.units as u

__all__ = [
    'MomentParameters',
    'HighOrderMomentParameters',
    'hillas_parameters',
    'HillasParameterizationError',
]

MomentParameters = namedtuple(
    "MomentParameters",
    "size,cen_x,cen_y,length,width,r,phi,psi,miss"
)
"""Shower moment parameters up to second order.

See also
--------
HighOrderMomentParameters, hillas_parameters
"""

HighOrderMomentParameters = namedtuple(
    "HighOrderMomentParameters",
    "Skewness, Kurtosis"
)
"""Shower moment parameters of third and fourth order.

See also
--------
MomentParameters, hillas_parameters
"""


class HillasParameterizationError(RuntimeError):
    pass


def hillas_parameters_1(pix_x, pix_y, image, higherMoments=True, reCalcPix=True):
    """Compute Hillas parameters for a given shower image.

    Reference: Appendix of the Whipple Crab paper Weekes et al. (1998) /
    //MP: 1998 paper has no appendix with the parameters, means 1989 paper
    http://adsabs.harvard.edu/abs/1989ApJ...342..379W
    (corrected for some obvious typos)
    //MP: probably better to use Whipple Reynolds et al 1993 paper:
    http://adsabs.harvard.edu/abs/1993ApJ...404..206R

    Parameters
    ----------
    pix_x : array_like
        Pixel x-coordinate
    pix_y : array_like
        Pixel y-coordinate
    image : array_like
        Pixel values corresponding
    higherMoments : Boolean (default True)
        Calculate also higher moments
    reCalcPix : Boolean (default True)
        Recalculate the pixel higher multiples (e.g., if pixels move (!) or pixel list changes between calls)


    Returns
    -------
    hillas_parameters : `MomentParameters`
    """
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
    S_x4 = np.sum(image * (pix_x - mean_x) ** 4) / size
    S_y4 = np.sum(image * (pix_y - mean_y) ** 4) / size

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
    # Angle between ellipse major ax. and x-axis of camera.
    # Will be used for disp
    delta = np.arctan(a)
    b = mean_y - a * mean_x
    # Sin & Cos Will be used for calculating higher order image parameters
    cos_delta = 1 / np.sqrt(1 + a * a)
    sin_delta = a * cos_delta

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

    if not higherMoments:
        return MomentParameters(size=size, cen_x=mean_x, cen_y=mean_y,
                                length=length, width=width,
                                r=r, phi=phi, psi=delta, miss=miss)

    # Higher order moments
    sk = cos_delta * (pix_x - mean_x) + sin_delta * (pix_y - mean_y)

    skewness = ((np.sum(image * np.power(sk, 3)) / size) /
                ((np.sum(image * np.power(sk, 2)) / size) ** (3. / 2)))
    kurtosis = ((np.sum(image * np.power(sk, 4)) / size) /
                ((np.sum(image * np.power(sk, 2)) / size) ** 2))
    asym3 = (np.power(cos_delta, 3) * S_xxx
             + 3.0 * np.power(cos_delta, 2) * sin_delta * S_xxy
             + 3.0 * cos_delta * np.power(sin_delta, 2) * S_xyy
             + np.power(sin_delta, 3) * S_yyy)
    asym = - np.power(-asym3, 1. / 3) if (asym3 < 0.) \
        else np.power(asym3, 1. / 3)

    assert np.sign(skewness) == np.sign(asym)

    # another definition of assymetry
    # asym = (mean_x - pix_x[np.argmax(image)]) * cos_delta
    #      + (mean_y - pix_y[np.argmax(image)]) * sin_delta

    # # Compute azwidth by transforming to (p, q) coordinates
    # sin_theta = mean_y / r
    # cos_theta = mean_x / r
    # q = (mean_x - pix_x) * sin_theta + (pix_y - mean_y) * cos_theta
    # m_q = np.sum(image * q) / size
    # m_qq = np.sum(image * q * q) / size
    # azwidth_2 = m_qq - m_q * m_q
    # azwidth = np.sqrt(azwidth_2)

    return (MomentParameters(size=size,
                             cen_x=mean_x,
                             cen_y=mean_y,
                             length=length,
                             width=width,
                             r=r,
                             phi=phi,
                             psi=delta,
                             miss=miss),
            HighOrderMomentParameters(Skewness=skewness,
                                      Kurtosis=kurtosis))

def static_pix(pix_x, pix_y, higherMoments, reCalcPix):
    """Hold static variables for a given camera's pixel positions,
    #if the camera's pixel positions haven't changed since last call, and otherwise or
    if first call initializes them to the right values.

    Parameters
    ----------
    pix_x : array_like
        Pixel x-coordinate
    pix_y : array_like
        Pixel y-coordinate
    higherMoments : Boolean
        Calculate also higher moments
    reCalcPix : Boolean
        Recalculate the pixel higher multiples (e.g., if pixels move (!) or pixel list changes between calls)

    Returns
    -------
    Nothing, but keeps variables as attributes to the function, so acts like a static variable holder."""

    # If not called before or reCalcPix
    if (not hasattr(static_pix, "pixdata") or reCalcPix): # \
      #or not (np.array_equal(pix_x,static_xy.pix_x) and np.array_equal(pix_y,static_xy.pix_y)):
      #, or if the pixel positions have changed, but this adds 15% calculation time
        static_pix.pixdata = np.row_stack([pix_x,
                                           pix_y,
                                           pix_x * pix_x,
                                           pix_x * pix_y,
                                           pix_y * pix_y])

        if higherMoments:
            # Add higher order moments x3, x2y, xy2, y3, x4, x3y, x2y2, xy3, y4
            static_pix.pixdataHO = \
                np.row_stack([pix_x * static_pix.pixdata[2],
                              static_pix.pixdata[2] * pix_y,
                              pix_x * static_pix.pixdata[4],
                              pix_y * static_pix.pixdata[4],
                              static_pix.pixdata[2] * static_pix.pixdata[2],
                              static_pix.pixdata[2] * static_pix.pixdata[3],
                              static_pix.pixdata[2] * static_pix.pixdata[4],
                              static_pix.pixdata[3] * static_pix.pixdata[4],
                              static_pix.pixdata[4] * static_pix.pixdata[4]])


def hillas_parameters_2(pix_x, pix_y, image, higherMoments=True, reCalcPix=True):
    """Compute Hillas parameters for a given shower image.

    Alternate implementation of `hillas_parameters` ...
    in the end we'll just keep one, but we're using Hillas parameter
    computation as an example for performance checks.

    Parameters
    ----------
    pix_x : array_like
        Pixel x-coordinate
    pix_y : array_like
        Pixel y-coordinate
    image : array_like
        Pixel values corresponding
    higherMoments : Boolean (default True)
        Calculate also higher moments
    reCalcPix : Boolean (default True)
        Recalculate the pixel higher multiples (e.g., if pixels move (!) or pixel list changes between calls)

    Returns
    -------
    hillas_parameters : `MomentParameters`
    """

    if type(pix_x)==Quantity:
        unit = pix_x.unit()
        assert pix_x.unit() == pix_y.unit()
    else:
        unit = 1.0

    pix_x = Quantity(np.asanyarray(pix_x, dtype=np.float64)).value
    pix_y = Quantity(np.asanyarray(pix_y, dtype=np.float64)).value
    image = np.asanyarray(image)

    assert pix_x.shape == image.shape
    assert pix_y.shape == image.shape

    size = image.sum()

    if size == 0.0:
        raise (HillasParameterizationError("Empty pixels! Cannot calculate image parameters. Exiting..."))

    #call static_xy to initialize the "static variables"
    #actually, would be nice to just call this if we know the pixel positions have changed
    static_pix(pix_x, pix_y, higherMoments, reCalcPix)

    # Compute image moments (done in a bit faster way, but putting all
    # into one 2D array, where each row will be summed to calculate a
    # moment) However, this doesn't avoid a temporary created for the
    # 2D array

    # momdata = np.row_stack([pix_x,
    #                         pix_y,
    #                         pix_x * pix_x,
    #                         pix_y * pix_y,
    #                         pix_x * pix_y]) * image
    momdata = static_pix.pixdata * image
    moms = momdata.sum(axis=1) / size

    if higherMoments:
        momdataHO = static_pix.pixdataHO * image
        momsHO = momdataHO.sum(axis=1) / size

    # give the moms values comprehensible names
    xm, ym, x2m, xym, y2m = moms
    if higherMoments:
        x3m, x2ym, xy2m, y3m, x4m, x3ym, x2y2m, xy3m, y4m = momsHO

    # intermediate variables (could be avoided if compiler which understands powers, etc)
    xm2 = xm * xm
    ym2 = ym * ym
    xmym = xm * ym

    # calculate variances

    vx2 = x2m - xm2
    vy2 = y2m - ym2
    vxy = xym - xmym

    if higherMoments:
        vx3 = x3m - 3.0 * xm * x2m + 2.0 * xm2 * xm
        vx2y = x2ym - x2m * ym - 2.0 * xym * xm + 2.0 * xm2 * ym
        vxy2 = xy2m - y2m * xm - 2.0 * xym * ym + 2.0 * xm * ym2
        vy3 = y3m - 3.0 * ym * y2m + 2.0 * ym2 * ym

    # polar coordinates of centroid

    rr = np.sqrt(xm2 + ym2)  # could use hypot(xm, ym), but already have squares
    phi = np.arctan2(ym, xm)

    # common factors:

    dd = vy2 - vx2
    zz = np.hypot(dd, 2.0 * vxy) # for simpler formulae for length & width suggested CA 901019

    # shower shape parameters

    length = np.sqrt((vx2 + vy2 + zz)/ 2.0)
    width  = np.sqrt((vx2 + vy2 - zz)/2.0)
    # azwidth = np.sqrt(x2m + y2m - zz) #Hillas Azwidth not used anymore
    # d = y2m - x2m
    # z = np.sqrt(d * d + 4 * xym * xym)
    # akwidth = np.sqrt((x2m + y2m - z) / 2.0) # Akerlof azwidth (910112) not used anymore either

    # miss, simpler formula for miss introduced CA, 901101; revised MP 910112

    uu = 1.0 + dd / zz
    vv = 2.0 - uu
    miss = np.sqrt((uu * xm2 + vv * ym2) / 2.0
                   - xmym * 2.0 * vxy / zz)
    # maybe for case zz = 0, put miss = dist?

    # rotation angle of ellipse relative to centroid
    tanpsi_numer = (dd + zz) * ym + (2.0 * vxy * xm)
    tanpsi_denom = (2.0 * vxy * ym) - (dd - zz) * xm

    psi = (np.arctan2(tanpsi_numer, tanpsi_denom)) * u.rad

    if not higherMoments:
        return MomentParameters(size=size, cen_x=xm*unit, cen_y=ym*unit,
                                length=length*unit, width=width,
                                r=rr*unit, phi=phi, psi=psi.to(u.rad), miss=miss*unit)

    # -- Asymmetry and other higher moments
    if length != 0.0:
        vx4 = x4m - 4.0 * xm * x3m + 6.0 * xm2 * x2m - 3.0 * xm2 * xm2
        vx3y = x3ym - 3.0 * xm * x2ym  + 3.0 * xm2 * xym - x3m * ym \
                    + 3.0 * x2m * xmym - 3.0 * xm2 * xmym
        vx2y2 = x2y2m - 2.0 * ym * x2ym + x2m * ym2 \
                      - 2.0 * xm * xy2m + 4.0 * xym * xmym + xm2 * y2m - 3.0 * xm2 * ym2
        vxy3 = xy3m - 3.0 * ym * xy2m  + 3.0 * ym2 * xym - y3m * xm \
                    + 3.0 * y2m * xmym - 3.0 * ym2 * xmym
        vy4 = y4m - 4.0 * ym * y3m + 6.0 * ym2 * y2m - 3.0 * ym2 * ym2

        hyp = np.hypot(tanpsi_numer,tanpsi_denom)
        if hyp != 0.:
            cpsi = tanpsi_denom / hyp
            spsi = tanpsi_numer / hyp
        else:
            cpsi = 1.
            spsi = 0.

        cpsi2 = cpsi * cpsi
        spsi2 = spsi * spsi
        cspsi = cpsi * spsi
        sk3bylen3 = (vx3 * cpsi*cpsi2 +
                     3.0 * vx2y * cpsi2 * spsi +
                     3.0 * vxy2 * cpsi * spsi2 +
                     vy3 * spsi*spsi2)
        asym = np.copysign(np.power(np.abs(sk3bylen3),1./3.), sk3bylen3) / length
        skewness = asym*asym*asym  # for MP's asym... (not for MAGIC asym!)
        # Kurtosis
        kurt = (vx4 * cpsi2 * cpsi2 +
                    4.0 * vx3y * cpsi2 * cspsi +
                    6.0 * vx2y2 * cpsi2 * spsi2 +
                    4.0 * vxy3 * cspsi * spsi2 +
                    vy4 * spsi2 * spsi2)
        kurtosis = kurt / (length*length*length*length)
    else:  # Skip Higher Moments
        psi = 0.0
        skewness = 0.0
        kurtosis = 0.0

    return MomentParameters(size=size, cen_x=xm*unit, cen_y=ym*unit,
                            length=length*unit, width=width, r=rr*unit, phi=phi, psi=psi.to(u.deg), miss=miss*unit), \
           HighOrderMomentParameters(Skewness=skewness, Kurtosis=kurtosis)


def hillas_parameters_3(pix_x, pix_y, image, higherMoments=True, reCalcPix=True):
    """Compute Hillas parameters for a given shower image.

    MP: probably better to use Whipple Reynolds et al 1993 paper:
    http://adsabs.harvard.edu/abs/1993ApJ...404..206R
    which should be the same as one of my ICRC 1991 papers and my thesis.

    Parameters
    ----------
    pix_x : array_like
        Pixel x-coordinate
    pix_y : array_like
        Pixel y-coordinate
    image : array_like
        Pixel values corresponding
    higherMoments : Boolean (default True)
        Calculate also higher moments
    reCalcPix : Boolean (default True)
        Recalculate the pixel higher multiples (e.g., if pixels move (!) or pixel list changes between calls)

    Returns
    -------
    hillas_parameters : `MomentParameters`
    """

    if type(pix_x)==Quantity:
        unit = pix_x.unit()
        assert pix_x.unit() == pix_y.unit()
    else:
        unit = 1.0

    pix_x = Quantity(np.asanyarray(pix_x, dtype=np.float64)).value
    pix_y = Quantity(np.asanyarray(pix_y, dtype=np.float64)).value
    image = np.asanyarray(image, dtype=np.float64)
    assert pix_x.shape == image.shape
    assert pix_y.shape == image.shape

    # Code to interface with historic version
    wxdeg = pix_x
    wydeg = pix_y
    event = image

    # Code below is from the historical Whipple routine, with original comments
    # MJL is Mark Lang, MP is Michael Punch, GV is Giuseppe Vacanti, CA is Carl Akerlof, RCL is Dick Lamb,

    # Translated from ForTran to c with "fable": https://github.com/ctessum/fable-go
    # and from c to Python with "cpp2python:  https://github.com/andreikop/cpp2python , then edited by hand
    # C***********************************************************************
    # C                              HILLAKPAR                               *
    # C***********************************************************************
    # C-- This version also calculates the asymmetry of the image.
    # C-- Use of Akerlof Azwidth (Akwidth?) implemented MP 910112
    # C-- Simpler Miss formula suggested CA, MP, 901101
    # C-- Simpler Width and Length formulae suggested CA, 901017
    # C-- Yet another little bug fixed by MP 900418- Generalize the case of
    # C--      the horizontal line.
    # C-- Bug fixed by RCL 900307-The case of the horizontal line image was
    # C--     never considered.
    # C-- Bug fixed by GV 900215
    # C** This version takes events in WHIPPLE format and parameterises them.
    # C**    M. Punch ,900105
    # C-- G. Vacanti introduces the common statement: coordinates are
    # C-- computed only in the main program.
    # C-- Modified by M. Punch to make it faster April, 1989
    # C-- mjl 10 dec 87
    # C--
    # -- routine to calculate the six hillas iarameters

    sumsig , sumxsig, sumysig, sumx2sig, sumy2sig, sumxysig, sumx3sig, sumx2ysig, sumxy2sig, sumy3sig = np.zeros(10)

    for i in range(np.size(event)):
        if event[i] != 0.0:
            wxbyev = wxdeg[i] * event[i]
            wybyev = wydeg[i] * event[i]
            sumsig += event[i]
            sumxsig += wxbyev
            sumx2sig += wxdeg[i] * wxbyev
            sumysig += wybyev
            sumy2sig += wydeg[i] * wybyev
            sumxysig += wxdeg[i] * wybyev
            sumx3sig += wxdeg[i] * wxdeg[i] * wxbyev
            sumx2ysig += wxdeg[i] * wxdeg[i] * wybyev
            sumxy2sig += wxdeg[i] * wydeg[i] * wybyev
            sumy3sig += wydeg[i] * wydeg[i] * wybyev

    if sumsig == 0.0:
        raise (HillasParameterizationError("Empty pixels! Cannot calculate image parameters. Exiting..."))

    xm = sumxsig / sumsig
    x2m = sumx2sig / sumsig
    ym = sumysig / sumsig
    y2m = sumy2sig / sumsig
    xym = sumxysig / sumsig

    xm2 = xm * xm
    ym2 = ym * ym
    xmym = xm * ym

    vx2 = x2m - xm2
    vy2 = y2m - ym2
    vxy = xym - xmym

    if higherMoments:
        x3m = sumx3sig / sumsig
        x2ym = sumx2ysig / sumsig
        xy2m = sumxy2sig / sumsig
        y3m = sumy3sig / sumsig

        vx3 = x3m - 3.0 * xm * x2m + 2.0 * xm2 * xm
        vx2y = x2ym - x2m * ym - 2.0 * xym * xm + 2.0 * xm2 * ym
        vxy2 = xy2m - y2m * xm - 2.0 * xym * ym + 2.0 * xm * ym2
        vy3 = y3m - 3.0 * ym * y2m + 2.0 * ym2 * ym

    d = vy2 - vx2
    dist = np.sqrt(xm2 + ym2)
    phi = np.arctan2(ym, xm)

    # -- simpler formulae for length & width suggested CA 901019
    z = np.sqrt(d * d + 4.0 * vxy * vxy)
    length = np.sqrt((vx2 + vy2 + z) / 2.0)
    width = np.sqrt((vy2 + vx2 - z) / 2.0)

    # -- simpler formula for miss introduced CA, 901101
    # -- revised MP 910112
    if z == 0.0:
        miss = dist
    else:
        u = 1 + d / z
        v = 2 - u
        miss = np.sqrt((u * xm2 + v * ym2) / 2.0 - xmym * (2.0 * vxy / z))

    # Code to de-interface with historical code
    size = sumsig
    m_x = xm*unit
    m_y = ym*unit
    length = length*unit
    width = width*unit
    r = dist*unit

    psi = np.arctan2((d + z) * ym + 2.0 * vxy * xm, 2.0 * vxy * ym - (d - z) * xm)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    if not higherMoments:
        return MomentParameters(size=size, cen_x=m_x, cen_y=m_y, length=length, width=width,
                                r=r, phi=phi, psi=psi, miss=miss)

    # -- Asymmetry
    if length == 0.0:
        asymm = 0.0
    else:
        asymm = (vx3 * np.power(cpsi,3) +
                 3.0 * vx2y * spsi * np.power(cpsi,2) + 3.0 * vxy2 * cpsi * np.power(spsi,2) +
                 vy3 * np.power(spsi,3))
        asymm = np.copysign(np.exp(np.log(np.abs(asymm)) / 3.0), asymm) / length

    # # -- Akerlof azwidth now used, 910112
    # d = y2m - x2m
    # z = np.sqrt(d * d + 4 * xym * xym)
    # azwidth = np.sqrt((x2m + y2m - z) / 2.0)
    #
    # isize = int(sumsig)

    # Code to de-interface with historical code
    skewness = asymm*asymm*asymm
    kurtosis = np.nan

    return MomentParameters(size=size, cen_x=m_x, cen_y=m_y, length=length, width=width,
                            r=r, phi=phi, psi=psi, miss=miss), \
           HighOrderMomentParameters(Skewness=skewness, Kurtosis=kurtosis)

def static_xy(pix_x,pix_y, higherMoments, reCalcPix):
    """Hold static variables for a given camera's pixel positions,
    #if the camera's pixel positions haven't changed since last call, and otherwise or
    if first call initializes them to the right values.

    Parameters
    ----------
    pix_x : array_like
        Pixel x-coordinate
    pix_y : array_like
        Pixel y-coordinate
    higherMoments : Boolean
        Calculate also higher moments
    reCalcPix : Boolean
        Recalculate the pixel higher multiples (e.g., if pixels move (!) or pixel list changes between calls)

    Returns
    -------
    Nothing, but keeps variables as attributes to the function, so acts like a static variable holder."""

    # If not called before or reCalcPix
    if (not hasattr(static_xy, "pix_x") or reCalcPix): # \
      #or not (np.array_equal(pix_x,static_xy.pix_x) and np.array_equal(pix_y,static_xy.pix_y)):
      #, or if the pixel positions have changed, but this adds 15% calculation time
        static_xy.pix_x = pix_x
        static_xy.pix_y = pix_y
        static_xy.pix_x2 = pix_x * pix_x
        static_xy.pix_y2 = pix_y * pix_y
        static_xy.pix_xy = pix_x * pix_y
        static_xy.pix_x3 = static_xy.pix_x2 * pix_x
        static_xy.pix_x2y = static_xy.pix_x2 * pix_y
        static_xy.pix_xy2 = pix_x * static_xy.pix_y2
        if higherMoments:
            static_xy.pix_y3 = pix_y * static_xy.pix_y2
            static_xy.pix_x4 = static_xy.pix_x3 * pix_x
            static_xy.pix_x3y = static_xy.pix_x3 * pix_y
            static_xy.pix_x2y2 = static_xy.pix_x2 * static_xy.pix_y2
            static_xy.pix_xy3 = pix_x * static_xy.pix_y3
            static_xy.pix_y4 = static_xy.pix_y3 * pix_y

def hillas_parameters_4(pix_x, pix_y, image, higherMoments=True, reCalcPix=True):
    """Compute Hillas parameters for a given shower image.

    As for hillas_parameters_3 (old Whipple Fortran code), but more Pythonized

    MP: Parameters calculated as Whipple Reynolds et al 1993 paper:
    http://adsabs.harvard.edu/abs/1993ApJ...404..206R
    which should be the same as one of my ICRC 1991 papers and my thesis.

    Parameters
    ----------
    pix_x : array_like
        Pixel x-coordinate
    pix_y : array_like
        Pixel y-coordinate
    image : array_like
        Pixel values corresponding
    higherMoments : Boolean (default True)
        Calculate also higher moments
    reCalcPix : Boolean (default True)
        Recalculate the pixel higher multiples (e.g., if pixels move (!) or pixel list changes between calls)

    Returns
    -------
    hillas_parameters : `MomentParameters`
    """


    if type(pix_x)==Quantity:
        unit = pix_x.unit()
        assert pix_x.unit() == pix_y.unit()
    else:
        unit = 1.0
    # MP: Actually, I don't know why we need to strip the units... shouldn' the calculations all work with them?

    pix_x = Quantity(np.asanyarray(pix_x, dtype=np.float64)).value
    pix_y = Quantity(np.asanyarray(pix_y, dtype=np.float64)).value
    image = np.asanyarray(image, dtype=np.float64)
    assert pix_x.shape == image.shape
    assert pix_y.shape == image.shape

    sumsig , sumxsig, sumysig, sumx2sig, sumy2sig, sumxysig, sumx3sig, sumx2ysig, sumxy2sig, sumy3sig = np.zeros(10)

    # Call static_xy to initialize the "static variables"
    # Actually, would be nice to just call this if we know the pixel positions have changed
    static_xy(pix_x,pix_y,higherMoments,reCalcPix)

    sumsig = image.sum()
    sumxsig = (image * pix_x).sum()
    sumysig = (image * pix_y).sum()
    sumx2sig = (image * static_xy.pix_x2).sum()
    sumy2sig = (image * static_xy.pix_y2).sum()
    sumxysig = (image * static_xy.pix_xy).sum()

    if higherMoments:
        sumx3sig = (image * static_xy.pix_x3).sum()
        sumx2ysig = (image * static_xy.pix_x2y).sum()
        sumxy2sig = (image * static_xy.pix_xy2).sum()
        sumy3sig = (image * static_xy.pix_y3).sum()

        sumx4sig = (image * static_xy.pix_x4).sum()
        sumx3ysig = (image * static_xy.pix_x3y).sum()
        sumx2y2sig = (image * static_xy.pix_x2y2).sum()
        sumxy3sig = (image * static_xy.pix_xy3).sum()
        sumy4sig = (image * static_xy.pix_y4).sum()

    if sumsig == 0.0:
        raise (HillasParameterizationError("Empty pixels! Cannot calculate image parameters. Exiting..."))

    xm   = sumxsig   / sumsig
    ym   = sumysig   / sumsig
    x2m  = sumx2sig  / sumsig
    y2m  = sumy2sig  / sumsig
    xym  = sumxysig  / sumsig

    if higherMoments:
        x3m  = sumx3sig  / sumsig
        x2ym = sumx2ysig / sumsig
        xy2m = sumxy2sig / sumsig
        y3m  = sumy3sig  / sumsig

        x4m  = sumx4sig  / sumsig
        x3ym = sumx3ysig / sumsig
        x2y2m = sumx2y2sig / sumsig
        xy3m = sumxy3sig / sumsig
        y4m  = sumy4sig  / sumsig

    # Doing this should be same as above, but its 4us slower !?
    #(xm, ym, x2m, y2m, xym, x3m, x2ym, xy2m, y3m) = \
    #    (sumxsig, sumysig, sumx2sig, sumy2sig, sumxysig, sumx3sig, sumx2ysig, sumxy2sig, sumy3sig) / sumsig

    xm2 = xm * xm
    ym2 = ym * ym
    xmym = xm * ym

    vx2 = x2m - xm2
    vy2 = y2m - ym2
    vxy = xym - xmym

    if higherMoments:
        vx3 = x3m - 3.0 * xm * x2m + 2.0 * xm2 * xm
        vx2y = x2ym - x2m * ym - 2.0 * xym * xm + 2.0 * xm2 * ym
        vxy2 = xy2m - y2m * xm - 2.0 * xym * ym + 2.0 * xm * ym2
        vy3 = y3m - 3.0 * ym * y2m + 2.0 * ym2 * ym
        # print("vs 3th: x3, x2y, xy2, y4", vx3, vx2y, vxy2, vy3)
        # print("cross check vx3",(image*(pix_x-xm)**3).sum()/sumsig)
        # print("cross check vx2y",(image*(pix_x-xm)**2*(pix_y-ym)).sum()/sumsig)
        # print("cross check vxy2",(image*(pix_x-xm)*(pix_y-ym)**2).sum()/sumsig)
        # print("cross check vy3",(image*(pix_y-ym)**3).sum()/sumsig)

    d = vy2 - vx2
    dist = np.sqrt(xm2 + ym2) #could use hypot(xm,ym), but already have squares
    phi = np.arctan2(ym, xm)

    # -- simpler formulae for length & width suggested CA 901019
    z = np.hypot(d, 2.0 * vxy)
    length = np.sqrt((vx2 + vy2 + z) / 2.0)
    width = np.sqrt((vy2 + vx2 - z) / 2.0)

    # -- simpler formula for miss introduced CA, 901101
    # -- revised MP 910112
    if z == 0.0:
        miss = dist
    else:
        u = 1 + d / z
        v = 2 - u
        miss = np.sqrt((u * xm2 + v * ym2) / 2.0 - xmym * (2.0 * vxy / z))

    #Change to faster caluclation of psi and avoid inaccuracy for hyp
    #psi = np.arctan2((d + z) * ym + 2.0 * vxy * xm, 2.0 *vxy * ym - (d - z) * xm)
    #hyp = np.sqrt(2 * z * (z + d))  #! should be simplification of sqrt((d+z)**2+(2*vxy)**2 ... but not accurate!
    #hyp = np.hypot(d + z,2 * vxy)
    #psi = np.arctan2(d + z, 2 * vxy)
    #cpsi = np.cos(psi)
    #spsi = np.sin(psi)
    tanpsi_numer = (d + z) * ym + 2.0 * vxy * xm
    tanpsi_denom = 2.0 *vxy * ym - (d - z) * xm
    psi = np.arctan2(tanpsi_numer,tanpsi_denom)

    # Code to de-interface with historical code
    size = sumsig
    m_x = xm*unit
    m_y = ym*unit
    length = length*unit
    width = width*unit
    r = dist*unit
    psi = psi

    if not higherMoments:
        return MomentParameters(size=size, cen_x=m_x, cen_y=m_y,
                                length=length, width=width, r=r, phi=phi, psi=psi, miss=miss)

    # Note, "skewness" is the same as the Whipple/MP "asymmetry^3", which is fine.
    # ... and also, Whipple/MP "asymmetry" * "length" = MAGIC "asymmetry"
    # ... so, MAGIC "asymmetry" = MAGIC "skewness"^(1/3) * "length"
    # I don't know what MAGIC's "asymmetry" is supposed to be.

    # -- Asymmetry and other higher moments
    if length != 0.0:
        vx4 = x4m - 4.0 * xm * x3m + 6.0 * xm2 * x2m - 3.0 * xm2 * xm2
        vx3y = x3ym - 3.0 * xm * x2ym + 3.0 * xm2 * xym - x3m * ym \
                  + 3.0 * x2m * xmym - 3.0 * xm2 * xm * ym
        vx2y2 = x2y2m - 2.0 * ym * x2ym + x2m * ym2 \
                   - 2.0 * xm * xy2m + 4.0 * xym * xmym + xm2 * y2m - 3.0 * xm2 * ym2
        vxy3 = xy3m - 3.0 * ym * xy2m + 3.0 * ym2 * xym - y3m * xm \
                  + 3.0 * y2m * xmym - 3.0 * ym2 * ym * xm
        vy4 = y4m - 4.0 * ym * y3m + 6.0 * ym2 * y2m - 3.0 * ym2 * ym2
        # print("vs 4th: x4, x3y, x2y2, xy3, y4", vx4, vx3y, vx2y2, vxy3, vy4)
        # print("cross check vx4",(image*(pix_x-xm)**4).sum()/sumsig)
        # print("cross check vx3y",(image*(pix_x-xm)**3*(pix_y-ym)).sum()/sumsig)
        # print("cross check vx2y2",(image*(pix_x-xm)**2*(pix_y-ym)**2).sum()/sumsig)
        # print("cross check vxy3",(image*(pix_x-xm)*(pix_y-ym)**3).sum()/sumsig)
        # print("cross check vy4",(image*(pix_y-ym)**4).sum()/sumsig)

        hyp = np.hypot(tanpsi_numer, tanpsi_denom)
        if hyp != 0.:
            cpsi = tanpsi_denom / hyp
            spsi = tanpsi_numer / hyp
        else:
            cpsi = 1.
            spsi = 0.

        cpsi2 = cpsi * cpsi
        spsi2 = spsi * spsi
        cspsi = cpsi * spsi

        sk3bylen3 = (vx3 * cpsi*cpsi2 +
                    3.0 * vx2y * cpsi2 * spsi +
                    3.0 * vxy2 * cpsi * spsi2 +
                    vy3 * spsi*spsi2)
        asym = np.copysign(np.power(np.abs(sk3bylen3),1./3.), sk3bylen3) / length
        skewness = asym*asym*asym  # for MP's asym... (not for MAGIC asym!)

        # Kurtosis
        kurt = (vx4 * cpsi2 * cpsi2 +
                    4.0 * vx3y * cpsi2 * cspsi +
                    6.0 * vx2y2 * cpsi2 * spsi2 +
                    4.0 * vxy3 * cspsi * spsi2 +
                    vy4 * spsi2 * spsi2)
        kurtosis = kurt / (length*length*length*length)

    else:  # Skip Higher Moments
        asym = 0.0
        psi = 0.0
        skewness = 0.0
        kurtosis = 0.0

    # Azwidth not used anymore
    # # -- Akerlof azwidth now used, 910112
    # d = y2m - x2m
    # z = np.sqrt(d * d + 4 * xym * xym)
    # azwidth = np.sqrt((x2m + y2m - z) / 2.0)

    return MomentParameters(size=size, cen_x=m_x, cen_y=m_y, length=length, width=width, r=r, phi=phi, psi=psi,
                            miss=miss), HighOrderMomentParameters(Skewness=skewness, Kurtosis=kurtosis)

# use the 1 version by default. Version 2 has apparent differences.
hillas_parameters = hillas_parameters_1

