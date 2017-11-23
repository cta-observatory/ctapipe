import os

import numpy as np
from astropy.utils.decorators import deprecated
from astropy import units as u

from ctapipe import utils
from ctapipe.utils import linalg


@deprecated(0.1, "will be replaced with proper coord transform")
def pixel_position_to_direction(pix_x, pix_y, tel_phi, tel_theta, tel_foclen):
    """
    TODO replace with proper implementation
    calculates the direction vector of corresponding to a
    (x,y) position on the camera

    beta is the pixel's angular distance to the centre
    according to beta / tel_view = r / maxR
    alpha is the polar angle between the y-axis and the pixel
    to find the direction the pixel is looking at:

    - the pixel direction is set to the telescope direction
    - offset by beta towards up
    - rotated around the telescope direction by the angle alpha


    Parameters
    -----------
    pix_x, pix_y : ndarray
        lists of x and y positions on the camera
    tel_phi, tel_theta: astropy quantities
        two angles that describe the orientation of the telescope
    tel_foclen : astropy quantity
        focal length of the telescope

    Returns
    -------
    pix_dirs : ndarray
        shape (n,3) list of "direction vectors"
        corresponding to a position on the camera
    """

    pix_alpha = np.arctan2(pix_y, pix_x)

    pix_rho = (pix_x ** 2 + pix_y ** 2) ** .5

    pix_beta = pix_rho / tel_foclen * u.rad

    tel_dir = linalg.set_phi_theta(tel_phi, tel_theta)

    pix_dirs = []
    for a, b in zip(pix_alpha, pix_beta):
        pix_dir = linalg.set_phi_theta(tel_phi, tel_theta - b)

        pix_dir = linalg.rotate_around_axis(pix_dir, tel_dir, a)
        pix_dirs.append(pix_dir * u.dimless)

    return pix_dirs


def alt_to_theta(alt):
    """transforms altitude (angle from the horizon upwards) to theta (angle from z-axis)
    for simtel array coordinate systems
    """
    return (90 * u.deg - alt).to(alt.unit)


def az_to_phi(az):
    """transforms azimuth (angle from north towards east)
    to phi (angle from x-axis towards y-axis)
    for simtel array coordinate systems
    """
    return -az


def transform_pixel_position(x, y):
    """transforms the x and y coordinates on the camera plane so that they correspond to
    the view as if looked along the pointing direction of the telescope, i.e. y->up and
    x->right
    """
    return x, -y
