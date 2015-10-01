from astropy.coordinates import Angle
from numpy import cos, sin, array


def rotation_matrix_2d(angle):
    """construct a 2D rotation matrix as a numpy NDArray that rotates a
    vector clockwise. Angle should be any object that can be converted
    into an `astropy.coordinates.Angle`
    """
    psi = Angle(angle).rad
    return array([[cos(psi), -sin(psi)],
                  [sin(psi),  cos(psi)]])
