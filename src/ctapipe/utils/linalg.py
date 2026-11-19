"""Provide vector norms, normalization, angles, and planar rotation matrices."""

import numpy as np
from astropy.coordinates import Angle
from numpy import cos, sin

__all__ = ["rotation_matrix_2d", "length", "normalise", "angle"]


def rotation_matrix_2d(angle):
    """Construct a matrix for a clockwise rotation in two dimensions.

    The angle can be any value accepted by `astropy.coordinates.Angle`. The result is a
    NumPy array.
    """
    psi = Angle(angle).rad
    return np.array([[cos(psi), -sin(psi)], [sin(psi), cos(psi)]])


def length(vec):
    """Compute the Euclidean norm of an array.

    The norm is the square root of the inner product of the array with itself.
    """
    return vec.dot(vec) ** 0.5


def normalise(vec):
    """Sets the length of the vector to 1 without changing its direction

    Parameters
    ----------
    vec : ndarray
        Vector to normalize.

    Returns
    -------
    numpy array with the same direction but length of 1
    """
    try:
        return vec / length(vec)
    except ZeroDivisionError:
        return vec


def angle(v1, v2):
    """Compute the angle between two Cartesian vectors in radians.

    The cosine is clipped to the range from minus one to one before applying the inverse
    cosine.

    Parameters
    ----------
    v1 : ndarray
        First Cartesian vector.
    v2 : ndarray
        Second Cartesian vector.

    Returns
    -------
    float
        Angle between the two vectors in radians.
    """
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.arccos(np.clip(v1.dot(v2) / norm, -1.0, 1.0))
