from astropy.coordinates import Angle
from astropy import units as u

import numpy as np
from numpy import cos, sin, arctan2 as atan2, arccos as acos

__all__ = ['rotate_around_axis', 'rotation_matrix_2d', 'length', 'normalise',
           'angle', 'set_phi_theta', 'set_phi_theta_r']


def rotation_matrix_2d(angle):
    """construct a 2D rotation matrix as a numpy NDArray that rotates a
    vector clockwise. Angle should be any object that can be converted
    into an `astropy.coordinates.Angle`
    """
    psi = Angle(angle).rad
    return np.array([[cos(psi), -sin(psi)],
                     [sin(psi), cos(psi)]])


def rotate_around_axis(vec, axis, angle):
    """ rotates a vector around an axis by an angle
        creates a rotation matrix and multiplies
        the initial vector with it

    Parameters
    ----------
    vec  : length-3 numpy array
            3D vector to be rotated
    axis : length-3 numpy array
            axis around which the rotation is performed
    angle : astropy angle quantity or float
            angle (in rad if float) by which vec is rotated around axis

    Returns
    -------
    rotated numpy array
    """

    axis = normalise(axis)
    a = cos(angle / 2.0)
    b, c, d = -axis * sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rot_matrix = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                           [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                           [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    return vec.dot(rot_matrix)


def length(vec):
    """ returns the length/norm of a numpy array
        as the square root of the inner product with itself
    """
    return vec.dot(vec)**.5


def normalise(vec):
    """ Sets the length of the vector to 1
        without changing its direction

    Parameters
    ----------
    vec : numpy array

    Returns
    -------
    numpy array with the same direction but length of 1
    """
    try:
        return vec / length(vec)
    except ZeroDivisionError:
        return vec


def angle(v1, v2):
    """ computes the angle between two vectors
        assuming carthesian coordinates

    Parameters
    ----------
    v1 : numpy array
    v2 : numpy array

    Returns
    -------
    the angle between v1 and v2 as a dimensioned astropy quantity
    """
    return acos(np.clip(v1.dot(v2) / (length(v1) * length(v2)), -1.0, 1.0))


def set_phi_theta_r(phi, theta, r=1):
    """ sets a 3D vector according to the given angles

    Parameters
    ----------
    phi : astropy.Quantity
    theta : astropy.Quantity
    r : (optional)
        the length of the vector
        can have a unit, doesn't have to

    Returns
    -------
    numpy array with the given direction and length
    """
    return np.array([sin(theta) * cos(phi),
                     sin(theta) * sin(phi),
                     cos(theta)]) * r


""" simple alias for set_phi_theta_r """
set_phi_theta = set_phi_theta_r


def get_phi_theta(vec):
    """ returns a tupel of the phi and theta angles of the given vector
    """
    try:
        return (atan2(vec[1], vec[0]), acos(np.clip(vec[2] / length(vec), -1, 1))) * u.rad
    except ValueError:
        return (0, 0)
