# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Hillas shower parametrization.

TODO:
-----

- Should have a separate function or option to compute 3rd order
  moments + asymmetry (which are not always needed)

- remove alpha calculation (which is only about (0,0), and make a get
  alpha function that does it from an arbitrary point given a
  pre-computed list of parameters

"""
import numpy as np
from astropy.units import Quantity
from collections import namedtuple
import astropy.units as u


__all__ = [
    'MomentParameters',
    'HighOrderMomentParameters',
    'hillas_parameters',
]


MomentParameters = namedtuple(
    "MomentParameters",
    "size,cen_x,cen_y,length,width,r,phi,psi,miss"
)
"""Shower moment parameters up to second order.

See also
--------
HighOrderMomentParameters, hillas_parameters, hillas_parameters_2
"""

HighOrderMomentParameters = namedtuple(
    "HighOrderMomentParameters",
    "skewness,kurtosis,asymmetry"
)
"""Shower moment parameters of third order.

See also
--------
MomentParameters, hillas_parameters, hillas_parameters_2
"""


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
    hillas_parameters : `MomentParameters`
    """
    pix_x = Quantity(np.asanyarray(pix_x, dtype=np.float64)).value
    pix_y = Quantity(np.asanyarray(pix_y, dtype=np.float64)).value
    image = np.asanyarray(image, dtype=np.float64)
    assert pix_x.shape == image.shape
    assert pix_y.shape == image.shape

    # Compute image moments
    _s = np.sum(image)
    m_x = np.sum(image * pix_x) / _s
    m_y = np.sum(image * pix_y) / _s
    m_xx = np.sum(image * pix_x * pix_x) / _s  # note: typo in paper
    m_yy = np.sum(image * pix_y * pix_y) / _s
    m_xy = np.sum(image * pix_x * pix_y) / _s  # note: typo in paper

    # Compute major axis line representation y = a * x + b
    S_xx = m_xx - m_x * m_x
    S_yy = m_yy - m_y * m_y
    S_xy = m_xy - m_x * m_y
    d = S_yy - S_xx
    temp = d * d + 4 * S_xy * S_xy
    a = (d + np.sqrt(temp)) / (2 * S_xy)
    b = m_y - a * m_x

    # Compute Hillas parameters
    width_2 = (S_yy + a * a * S_xx - 2 * a * S_xy) / (1 + a * a)
    width = np.sqrt(width_2)
    length_2 = (S_xx + a * a * S_yy + 2 * a * S_xy) / (1 + a * a)
    length = np.sqrt(length_2)
    miss = np.abs(b / (1 + a * a))
    r = np.sqrt(m_x * m_x + m_y * m_y)
    phi = np.arctan2(m_y, m_x)

    # Compute azwidth by transforming to (p, q) coordinates
    sin_theta = m_y / r
    cos_theta = m_x / r
    q = (m_x - pix_x) * sin_theta + (pix_y - m_y) * cos_theta
    m_q = np.sum(image * q) / _s
    m_qq = np.sum(image * q * q) / _s
    azwidth_2 = m_qq - m_q * m_q
    azwidth = np.sqrt(azwidth_2)

    return MomentParameters(size=_s, cen_x=m_x, cen_y=m_y, length=length,
                            width=width, r=r, phi=phi, psi=None, miss=miss)


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
    hillas_parameters : `MomentParameters`
    """
    unit = pix_x.unit

    pix_x = Quantity(np.asanyarray(pix_x, dtype=np.float64)).value
    pix_y = Quantity(np.asanyarray(pix_y, dtype=np.float64)).value

    assert pix_x.shape == image.shape
    assert pix_y.shape == image.shape

    # Compute image moments (done in a bit faster way, but putting all
    # into one 2D array, where each row will be summed to calculate a
    # moment) However, this doesn't avoid a temporary created for the
    # 2D array

    size = image.sum()
    momdata = np.row_stack([pix_x,
                            pix_y,
                            pix_x * pix_x,
                            pix_y * pix_y,
                            pix_x * pix_y]) * image

    moms = momdata.sum(axis=1) / size

    # calculate variances

    vx2 = moms[2] - moms[0] ** 2
    vy2 = moms[3] - moms[1] ** 2
    vxy = moms[4] - moms[0] * moms[1]

    # common factors:

    dd = vy2 - vx2
    zz = np.sqrt(dd ** 2 + 4.0 * vxy ** 2)

    # miss

    uu = 1.0 + dd / zz
    vv = 2.0 - uu
    miss = np.sqrt((uu * moms[0] ** 2 + vv * moms[1] ** 2) / 2.0
                   - moms[0] * moms[1] * 2.0 * vxy / zz)

    # shower shape parameters

    width = np.sqrt(vx2 + vy2 - zz)
    length = np.sqrt(vx2 + vy2 + zz)
    azwidth = np.sqrt(moms[2] + moms[3] - zz)

    # rotation angle of ellipse relative to centroid

    tanpsi_numer = (dd + zz) * moms[1] + 2.0 * vxy * moms[0]
    tanpsi_denom = (2 * vxy * moms[1]) - (dd - zz) * moms[0]
    psi = ((np.pi / 2.0) + np.arctan2(tanpsi_numer, tanpsi_denom))* u.rad

    # polar coordinates of centroid

    rr = np.hypot(moms[0], moms[1])
    phi = np.arctan2(moms[1], moms[0])

    return MomentParameters(size=size, cen_x=moms[0]*unit, cen_y=moms[1]*unit,
                            length=length*unit, width=width*unit, r=rr, phi=phi,
                            psi=psi.to(u.deg) , miss=miss*unit)


# use the 2 version by default
hillas_parameters = hillas_parameters_2
