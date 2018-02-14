# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: UTF-8 -*-
"""
Hillas-style moment-based shower image parametrization.
"""

from collections import namedtuple

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle
from astropy.units import Quantity

from ctapipe.instrument import CameraGeometry
from ..io.containers import HillasParametersContainer

__all__ = [
    'MomentParameters',
    'hillas_parameters',
    'hillas_parameters_1',
    'hillas_parameters_2',
    'hillas_parameters_3',
    'hillas_parameters_4',
    'HillasParameterizationError',
]

MomentParameters = namedtuple(
    "MomentParameters",
    "size,cen_x,cen_y,length,width,r,phi,psi,miss,skewness,kurtosis"
)


class HillasParameterizationError(RuntimeError):
    pass


def hillas_parameters_1(geom: CameraGeometry, image):
    """Compute Hillas parameters for a given shower image.

    Reference: Appendix of the Whipple Crab paper Weekes et al. (1998) /
    //MP: 1998 paper has no appendix with the parameters, means 1989 paper
    http://adsabs.harvard.edu/abs/1989ApJ...342..379W
    (corrected for some obvious typos)
    //MP: probably better to use Whipple Reynolds et al 1993 paper:
    http://adsabs.harvard.edu/abs/1993ApJ...404..206R

    Parameters
    ----------
    geom: ctapipe.instrument.CameraGeometry
        Camera corresponding to the image
    image : array_like
        Pixel values

    Returns
    -------
    hillas_parameters : `MomentParameters`
    """
    unit = Quantity(geom.pix_x).unit
    pix_x = Quantity(np.asanyarray(geom.pix_x, dtype=np.float64)).value
    pix_y = Quantity(np.asanyarray(geom.pix_y, dtype=np.float64)).value
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
    width_2 = ((S_yy + a * a * S_xx - 2 * a * S_xy) / (1 + a * a))
    length_2 = ((S_xx + a * a * S_yy + 2 * a * S_xy) / (1 + a * a))

    # Sanity check3:
    width = 0. if width_2 < 0. else np.sqrt(width_2)
    length = 0. if length_2 < 0. else np.sqrt(length_2)

    miss = np.abs(b / np.sqrt(1 + a * a))
    r = np.sqrt(mean_x * mean_x + mean_y * mean_y)
    phi = np.arctan2(mean_y, mean_x)

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
    asym = - np.power(-asym3, 1. / 3) if (asym3 < 0.) else np.power(asym3,
                                                                    1. / 3)

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

    return MomentParameters(size=size,
                            cen_x=mean_x * unit,
                            cen_y=mean_y * unit,
                            length=length * unit,
                            width=width * unit,
                            r=r * unit,
                            phi=Angle(phi * u.rad),
                            psi=Angle(delta * u.rad),
                            miss=miss * unit,
                            skewness=skewness,
                            kurtosis=kurtosis)


def hillas_parameters_2(geom: CameraGeometry, image):
    """Compute Hillas parameters for a given shower image.

    Alternate implementation of `hillas_parameters` ...
    in the end we'll just keep one, but we're using Hillas parameter
    computation as an example for performance checks.

    Parameters
    ----------
    geom: ctapipe.instrument.CameraGeometry
        Camera corresponding to the image
    image : array_like
        Pixel values


    Returns
    -------
    hillas_parameters : `MomentParameters`
    """

    if type(geom.pix_x) == Quantity:
        unit = geom.pix_x.unit
        assert geom.pix_x.unit == geom.pix_y.unit
    else:
        unit = 1.0

    pix_x = Quantity(np.asanyarray(geom.pix_x, dtype=np.float64)).value
    pix_y = Quantity(np.asanyarray(geom.pix_y, dtype=np.float64)).value
    image = np.asanyarray(image)

    assert pix_x.shape == image.shape
    assert pix_y.shape == image.shape

    size = image.sum()

    if size == 0.0:
        raise (HillasParameterizationError(("Empty pixels!"
                                            "Cannot calculate image parameters."
                                            "Exiting...")))

    pixdata = np.row_stack([pix_x,
                            pix_y,
                            pix_x * pix_x,
                            pix_x * pix_y,
                            pix_y * pix_y])

    # Compute image moments (done in a bit faster way, but putting all
    # into one 2D array, where each row will be summed to calculate a
    # moment) However, this doesn't avoid a temporary created for the
    # 2D array

    M = geom.pixel_moment_matrix

    momdata = (M @ image) / size

    # give the moms values comprehensible names

    (xm, ym, x2m, xym, y2m, x3m, x2ym, xy2m, y3m, x4m, x3ym, x2y2m, xy3m,
     y4m) = momdata

    # intermediate variables (could be avoided if compiler which understands powers, etc)
    xm2 = xm * xm
    ym2 = ym * ym
    xmym = xm * ym

    # calculate variances

    vx2 = x2m - xm2
    vy2 = y2m - ym2
    vxy = xym - xmym
    vx3 = x3m - 3.0 * xm * x2m + 2.0 * xm2 * xm
    vx2y = x2ym - x2m * ym - 2.0 * xym * xm + 2.0 * xm2 * ym
    vxy2 = xy2m - y2m * xm - 2.0 * xym * ym + 2.0 * xm * ym2
    vy3 = y3m - 3.0 * ym * y2m + 2.0 * ym2 * ym

    # polar coordinates of centroid

    rr = np.sqrt(xm2 + ym2)  # could use hypot(xm, ym), but already have squares
    phi = np.arctan2(ym, xm) * u.rad

    # common factors:

    dd = vy2 - vx2
    zz = np.hypot(dd, 2.0 * vxy)
    # for simpler formulae for length & width suggested CA 901019

    # shower shape parameters

    length = np.sqrt((vx2 + vy2 + zz) / 2.0)
    width = np.sqrt((vx2 + vy2 - zz) / 2.0)

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

    # -- Asymmetry and other higher moments
    if length != 0.0:
        vx4 = x4m - 4.0 * xm * x3m + 6.0 * xm2 * x2m - 3.0 * xm2 * xm2
        vx3y = (x3ym - 3.0 * xm * x2ym + 3.0 * xm2 * xym - x3m * ym
                + 3.0 * x2m * xmym - 3.0 * xm2 * xmym)
        vx2y2 = (x2y2m - 2.0 * ym * x2ym + x2m * ym2
                 - 2.0 * xm * xy2m + 4.0 * xym * xmym + xm2 * y2m
                 - 3.0 * xm2 * ym2)
        vxy3 = (xy3m - 3.0 * ym * xy2m + 3.0 * ym2 * xym - y3m * xm
                + 3.0 * y2m * xmym - 3.0 * ym2 * xmym)
        vy4 = y4m - 4.0 * ym * y3m + 6.0 * ym2 * y2m - 3.0 * ym2 * ym2

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
        sk3bylen3 = (vx3 * cpsi * cpsi2 +
                     3.0 * vx2y * cpsi2 * spsi +
                     3.0 * vxy2 * cpsi * spsi2 +
                     vy3 * spsi * spsi2)
        asym = np.copysign(np.power(np.abs(sk3bylen3), 1. / 3.),
                           sk3bylen3) / length
        skewness = asym * asym * asym  # for MP's asym... (not for MAGIC asym!)
        # Kurtosis
        kurt = (vx4 * cpsi2 * cpsi2 +
                4.0 * vx3y * cpsi2 * cspsi +
                6.0 * vx2y2 * cpsi2 * spsi2 +
                4.0 * vxy3 * cspsi * spsi2 +
                vy4 * spsi2 * spsi2)
        kurtosis = kurt / (length * length * length * length)
    else:  # Skip Higher Moments
        psi = 0.0 * u.rad
        skewness = 0.0
        kurtosis = 0.0

    return MomentParameters(size=size, cen_x=xm * unit, cen_y=ym * unit,
                            length=length * unit, width=width * unit,
                            r=rr * unit,
                            phi=Angle(phi), psi=Angle(psi),
                            miss=miss * unit, skewness=skewness,
                            kurtosis=kurtosis)


def hillas_parameters_3(geom: CameraGeometry, image):
    """Compute Hillas parameters for a given shower image.

    MP: probably better to use Whipple Reynolds et al 1993 paper:
    http://adsabs.harvard.edu/abs/1993ApJ...404..206R
    which should be the same as one of my ICRC 1991 papers and my thesis.

    Parameters
    ----------
    geom : CameraGeometry
        Geometry corresponding to the image
    image : array_like
        Pixel values corresponding

    Returns
    -------
    hillas_parameters : `MomentParameters`
    """

    if type(geom.pix_x) == Quantity:
        unit = geom.pix_x.unit
        assert geom.pix_x.unit == geom.pix_y.unit
    else:
        unit = 1.0

    pix_x = Quantity(np.asanyarray(geom.pix_x, dtype=np.float64)).value
    pix_y = Quantity(np.asanyarray(geom.pix_y, dtype=np.float64)).value

    # make sure they are numpy arrays so we can use numpy operations
    image = np.asanyarray(image, dtype=np.float64)

    assert pix_x.shape == image.shape
    assert pix_y.shape == image.shape

    # Code to interface with historic version
    wxdeg = pix_x
    wydeg = pix_y
    event = image

    # Code below is from the historical Whipple routine, with original comments
    # MJL is Mark Lang, MP is Michael Punch,
    # GV is Giuseppe Vacanti, CA is Carl Akerlof, RCL is Dick Lamb,

    # Translated from ForTran to c with "fable": https://github.com/ctessum/fable-go
    # and from c to Python with "cpp2python:
    # https://github.com/andreikop/cpp2python , then edited by hand
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

    (sumsig, sumxsig, sumysig, sumx2sig, sumy2sig,
     sumxysig, sumx3sig, sumx2ysig, sumxy2sig, sumy3sig) = np.zeros(10)

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
        raise (
            HillasParameterizationError(("Empty pixels! Cannot calculate image"
                                         "parameters. Exiting...")))

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
        uu = 1 + d / z
        vv = 2 - uu
        miss = np.sqrt((uu * xm2 + vv * ym2) / 2.0 - xmym * (2.0 * vxy / z))

    # Code to de-interface with historical code
    size = sumsig
    m_x = xm
    m_y = ym
    r = dist

    psi = np.arctan2((d + z) * ym + 2.0 * vxy * xm,
                     2.0 * vxy * ym - (d - z) * xm)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    # -- Asymmetry
    if length == 0.0:
        asymm = 0.0
    else:
        asymm = (vx3 * np.power(cpsi, 3) +
                 3.0 * vx2y * spsi * np.power(cpsi, 2) + 3.0 * vxy2 * cpsi *
                 np.power(spsi, 2) + vy3 * np.power(spsi, 3))
        asymm = np.copysign(np.exp(np.log(np.abs(asymm)) / 3.0), asymm) / length

    # # -- Akerlof azwidth now used, 910112
    # d = y2m - x2m
    # z = np.sqrt(d * d + 4 * xym * xym)
    # azwidth = np.sqrt((x2m + y2m - z) / 2.0)
    #
    # isize = int(sumsig)

    # Code to de-interface with historical code
    skewness = asymm * asymm * asymm
    kurtosis = np.nan

    return MomentParameters(size=size, cen_x=m_x * unit, cen_y=m_y * unit,
                            length=length * unit, width=width * unit,
                            r=r * unit, phi=Angle(phi * u.rad),
                            psi=Angle(psi * u.rad),
                            miss=miss * unit,
                            skewness=skewness, kurtosis=kurtosis)


def hillas_parameters_4(geom: CameraGeometry, image, container=False):
    """Compute Hillas parameters for a given shower image.

    As for hillas_parameters_3 (old Whipple Fortran code), but more Pythonized

    MP: Parameters calculated as Whipple Reynolds et al 1993 paper:
    http://adsabs.harvard.edu/abs/1993ApJ...404..206R
    which should be the same as one of my ICRC 1991 papers and my thesis.

    Parameters
    ----------
    geom: ctapipe.instrument.CameraGeometry
        Camera geometry
    image : array_like
        Pixel values
    container: bool
        return a HillasParametersContainer instead of a tuple

    Returns
    -------
    MomentParameters:
        tuple of hillas parameters
    HillasParametersContainer:
        container of hillas parametesr
    """

    if not isinstance(geom, CameraGeometry):
        raise ValueError("Hillas Parameters API has changed: hillas_parameters("
                         "geom, image). Please update your code")

    unit = geom.pix_x.unit

    # MP: Actually, I don't know why we need to strip the units... shouldn'
    # the calculations all work with them?'''

    pix_x = Quantity(np.asanyarray(geom.pix_x, dtype=np.float64)).value
    pix_y = Quantity(np.asanyarray(geom.pix_y, dtype=np.float64)).value
    image = np.asanyarray(image, dtype=np.float64)
    assert pix_x.shape == image.shape
    assert pix_y.shape == image.shape

    # Call static_xy to initialize the "static variables"
    # Actually, would be nice to just call this if we
    # know the pixel positions have changed

    sumsig = image.sum()

    if sumsig == 0.0:
        raise HillasParameterizationError("no signal to parametrize")

    M = geom.pixel_moment_matrix
    # Use a dot matrix approach that still supports masked arrays (@ does not)
    moms = np.ma.dot(M, image) / sumsig

    (xm, ym, x2m, xym, y2m, x3m, x2ym, xy2m, y3m, x4m, x3ym, x2y2m, xy3m,
     y4m) = moms

    # Doing this should be same as above, but its 4us slower !?
    # (xm, ym, x2m, y2m, xym, x3m, x2ym, xy2m, y3m) = \
    # (sumxsig, sumysig, sumx2sig, sumy2sig, sumxysig, sumx3sig,
    # sumx2ysig, sumxy2sig, sumy3sig) / sumsig

    xm2 = xm * xm
    ym2 = ym * ym
    xmym = xm * ym

    vx2 = x2m - xm2
    vy2 = y2m - ym2
    vxy = xym - xmym

    vx3 = x3m - 3.0 * xm * x2m + 2.0 * xm2 * xm
    vx2y = x2ym - x2m * ym - 2.0 * xym * xm + 2.0 * xm2 * ym
    vxy2 = xy2m - y2m * xm - 2.0 * xym * ym + 2.0 * xm * ym2
    vy3 = y3m - 3.0 * ym * y2m + 2.0 * ym2 * ym

    d = vy2 - vx2
    dist = np.sqrt(
        xm2 + ym2)  # could use hypot(xm,ym), but already have squares
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
        uu = 1 + d / z
        vv = 2 - uu
        miss = np.sqrt((uu * xm2 + vv * ym2) / 2.0 - xmym * (2.0 * vxy / z))

    tanpsi_numer = (d + z) * ym + 2.0 * vxy * xm
    tanpsi_denom = 2.0 * vxy * ym - (d - z) * xm
    psi = np.arctan2(tanpsi_numer, tanpsi_denom)

    # Code to de-interface with historical code
    size = sumsig
    m_x = xm
    m_y = ym
    length = length
    r = dist

    # Note, "skewness" is the same as the Whipple/MP "asymmetry^3", which is fine.
    # ... and also, Whipple/MP "asymmetry" * "length" = MAGIC "asymmetry"
    # ... so, MAGIC "asymmetry" = MAGIC "skewness"^(1/3) * "length"
    # I don't know what MAGIC's "asymmetry" is supposed to be.

    # -- Asymmetry and other higher moments
    if length != 0.0:
        vx4 = x4m - 4.0 * xm * x3m + 6.0 * xm2 * x2m - 3.0 * xm2 * xm2
        vx3y = (x3ym - 3.0 * xm * x2ym + 3.0 * xm2 * xym - x3m * ym
                + 3.0 * x2m * xmym - 3.0 * xm2 * xm * ym)
        vx2y2 = (x2y2m - 2.0 * ym * x2ym + x2m * ym2
                 - 2.0 * xm * xy2m + 4.0 * xym * xmym + xm2 * y2m
                 - 3.0 * xm2 * ym2)
        vxy3 = (xy3m - 3.0 * ym * xy2m + 3.0 * ym2 * xym - y3m * xm
                + 3.0 * y2m * xmym - 3.0 * ym2 * ym * xm)
        vy4 = y4m - 4.0 * ym * y3m + 6.0 * ym2 * y2m - 3.0 * ym2 * ym2
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

        sk3bylen3 = (vx3 * cpsi * cpsi2 +
                     3.0 * vx2y * cpsi2 * spsi +
                     3.0 * vxy2 * cpsi * spsi2 +
                     vy3 * spsi * spsi2)
        asym = np.copysign(np.power(np.abs(sk3bylen3), 1. / 3.),
                           sk3bylen3) / length
        skewness = asym * asym * asym  # for MP's asym... (not for MAGIC asym!)

        # Kurtosis
        kurt = (vx4 * cpsi2 * cpsi2 +
                4.0 * vx3y * cpsi2 * cspsi +
                6.0 * vx2y2 * cpsi2 * spsi2 +
                4.0 * vxy3 * cspsi * spsi2 +
                vy4 * spsi2 * spsi2)
        kurtosis = kurt / (length * length * length * length)

    else:  # Skip Higher Moments
        psi = 0.0
        skewness = 0.0
        kurtosis = 0.0

    if container:
        return HillasParametersContainer(x=m_x * unit, y=m_y * unit, r=r * unit,
                                         phi=Angle(phi * u.rad),
                                         intensity=size,
                                         length=length * unit,
                                         width=width * unit,
                                         psi=Angle(psi * u.rad),
                                         skewness=skewness,
                                         kurtosis=kurtosis)
    else:
        return MomentParameters(size=size, cen_x=m_x * unit, cen_y=m_y * unit,
                                length=length * unit, width=width * unit,
                                r=r * unit,
                                phi=Angle(phi * u.rad),
                                psi=Angle(psi * u.rad),
                                miss=miss * unit,
                                skewness=skewness, kurtosis=kurtosis)


# use the 4 version by default.
hillas_parameters = hillas_parameters_4
