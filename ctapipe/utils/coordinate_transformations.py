import os

import numpy as np
from astropy.utils.decorators import deprecated
from astropy import units as u

from ctapipe.utils import linalg


@deprecated(0.1, "will be replaced with proper coord transform")
def guess_pix_direction(pix_x, pix_y, tel_phi, tel_theta, tel_foclen):
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

    # the orientation of the camera (i.e. the pixel positions) needs to be corrected
    pix_x, pix_y = transf_pixel_position(pix_x, pix_y)
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
    return (90 * u.deg - alt).to(alt.unit)


def az_to_phi(az):
    return -az


def transf_array_position(x, y):
    return x, y


def transf_pixel_position(x, y):
    return x, -y


# functions to play through the different transformations
# set environment variables to control which transformation to use
# e.g. loop over the indices to bruteforce the correct set of transformations
def az_to_phi_debug(az):
    """azimuth is counted from north but phi from the x-axis.
    figure out where x is pointing by adding +-90° / 180° to `az`

    `az_deg` determines whether increases clock- or counter-clock-wise
    """
    az_deg = int(os.environ["AZDEG"])

    i = 0
    if int(os.environ["AZ"]) == i:
        return az_deg * az + 0 * u.deg
    i += 1
    if int(os.environ["AZ"]) == i:
        return az_deg * az + 90 * u.deg
    i += 1
    if int(os.environ["AZ"]) == i:
        return az_deg * az - 90 * u.deg
    i += 1
    if int(os.environ["AZ"]) == i:
        return az_deg * az + 180 * u.deg


def transf_array_position_debug(x, y):
    """find out where the x- and y-axes of the array are pointing by switching/flipping
    the coordinates of the telescope positions
    """

    i = 0
    if int(os.environ["PO"]) == i:
        return x, y
    i += 1
    if int(os.environ["PO"]) == i:
        return -x, y
    i += 1
    if int(os.environ["PO"]) == i:
        return x, -y
    i += 1
    if int(os.environ["PO"]) == i:
        return -x, -y

    i += 1
    if int(os.environ["PO"]) == i:
        return y, x
    i += 1
    if int(os.environ["PO"]) == i:
        return -y, -x
    i += 1
    if int(os.environ["PO"]) == i:
        return y, -x
    i += 1
    if int(os.environ["PO"]) == i:
        return -y, x


def transf_pixel_position_debug(x, y):
    """find out where the x- and y-axes of the camera are pointing by switching/flipping
    the coordinates of the pixel positions
    """
    i = 0
    if int(os.environ["PI"]) == i:
        return x, y
    i += 1
    if int(os.environ["PI"]) == i:
        return -x, y
    i += 1
    if int(os.environ["PI"]) == i:
        return x, -y
    i += 1
    if int(os.environ["PI"]) == i:
        return -x, -y

    i += 1
    if int(os.environ["PI"]) == i:
        return y, x
    i += 1
    if int(os.environ["PI"]) == i:
        return -y, -x
    i += 1
    if int(os.environ["PI"]) == i:
        return y, -x
    i += 1
    if int(os.environ["PI"]) == i:
        return -y, x
