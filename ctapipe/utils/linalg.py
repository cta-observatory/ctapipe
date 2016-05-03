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



def rotate_around_axis(vec, axis, angle):
    """ rotates @vec aroun @axis with @angle in radians
        creates a rotation matrix and calls the matrix 
        multiplication method
    
    Parameters
    ---------
    vec  : length-3 numpy array
            3D vector to be rotated
    axis : length-3 numpy array
            axis around which the rotation is performed
    angle : float
            angle by which @vec is rotated around @axis
    
    Result
    ------
    rotated numpy array
    """

    theta = np.asarray(angle.to(u.rad)/u.rad)
    axis = axis/(axis.dot(axis)**.5)
    a = cos(theta/2.0)
    b, c, d = -axis*sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    rot_matrix = np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                            [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                            [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
    return vec.dot(rot_matrix)

def length(vec):
    """ returns the length / norm of a numpy array
    """
    return vec.dot(vec)**.5
    
def normalise(vec):
    """ Sets the length of the vector to 1
        without changing its direction
        
    Parameter:
    ----------
    vec : numpy array
    
    Result:
    -------
    numpy array with the same direction but length of 1
    """
    return vec / length(vec)

def angle(v1, v2):
    """ computes the angle between two vectors
        assuming carthesian coordinates
        
    Parameters:
    -----------
    vec1 : numpy array
    vec2 : numpy array
    
    Result:
    -------
    the angle between vec1 and vec2 as a dimensioned astropy quantity
    """
    v1_u = normalise(v1)
    v2_u = normalise(v2)
    return acos(np.clip(v1_u.dot(v2_u), -1.0, 1.0))

def set_phi_theta_r(phi, theta, r=1):
    """ sets a 3D vector according to the given angles
    
    Parameters:
    ----------
    phi : astropy.Quantity
    theta : astropy.Quantity
    r : (optional)
        the length of the vector
        can have a unit, doesn't have to
        
    Result:
    -------
    numpy array with the given direction and length
    """
    return np.array([ sin(theta)*cos(phi),
                      sin(theta)*sin(phi),
                      cos(theta)         ])*r
""" simple alias for set_phi_theta_r with default (unitless) r argument """
set_phi_theta = lambda x, y: set_phi_theta_r(x,y)

def get_phi_theta(vec):
    """ returns a tupel of the phi and theta angles of the given vector
    """
    try:
        return ( atan2(vec[1], vec[0]), acos( np.clip(vec[2] / Length(vec), -1, 1) ) ) * u.rad
    except ValueError:
        return (0,0)

def distance(vec1, vec2):
    """ computes the distance between two vectors as the length of their difference
    
    Parameters:
    ----------
    vec1 : numpy array
    vec2 : numpy array
    
    Result:
    -------
    distance between the two vectors
    
    """
    return length(vec1 - vec2)
