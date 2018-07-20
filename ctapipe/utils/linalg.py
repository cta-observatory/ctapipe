from astropy.coordinates import Angle
from astropy import units as u

import numpy as np
from numpy import cos, sin, arctan2 as atan2, arccos as acos

__all__ = ['rotation_matrix_2d', 'length', 'normalise', 'angle']


def rotation_matrix_2d(angle):
    """construct a 2D rotation matrix as a numpy NDArray that rotates a
    vector clockwise. Angle should be any object that can be converted
    into an `astropy.coordinates.Angle`
    """
    psi = Angle(angle).rad
    return np.array([[cos(psi), -sin(psi)],
                     [sin(psi), cos(psi)]])



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
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.arccos(np.clip(v1.dot(v2) / norm, -1.0, 1.0))
