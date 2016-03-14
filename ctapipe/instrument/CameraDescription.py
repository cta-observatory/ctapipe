import astropy
from astropy import units as u
from astropy.table import Table
import numpy as np
from scipy.spatial import cKDTree as KDTree
from ctapipe.utils.linalg import rotation_matrix_2d

__all__ = ['Camera','npix_to_type','find_neighbor_pixels',
           'guess_camera_geometry','guess_camera_type',
           'make_rectangular_camera_geometry','rotate_camera']
    
class Camera:
    
    """`Camera` is a class that provides and gets all the information about
    the camera of a specific telescope."""
    
    def __init__(self,pix_id,pix_X,pix_Y,
                 pix_area,pix_type,pix_neighbors):
        """
        Parameters
        ----------
        self: type
            description
        pix_id: array (int)
            pixel ids of the camera of the telescope
        pix_posX: array with units
            position of each pixel (x-coordinate)
        pix_posY: array with units
            position of each pixel (y-coordinate)
        pix_area: array with units
            area of each pixel
        pix_type: string
            name of the pixel type (e.g. hexagonal)
        pix_neighbors: ndarray (int)
            nD-array with pixel IDs of neighboring
            pixels of the pixels (n=number of pixels)
        """
        self.pix_id = pix_id
        self.pix_X = pix_X
        self.pix_Y = pix_Y
        self.pix_area = pix_area
        self.pix_type = pix_type
        self.pix_neighbors = pix_neighbors
    
    @staticmethod
    def guess(cls):
        """
        Construct a `CameraGeometry` by guessing the appropriate quantities
        from a list of pixel positions.
        """
        return guess_camera_geometry(cls.pix_x, cls.pix_y)
    
    @staticmethod
    def rotate(cls,angle):
        return rotate_camera(angle,cls.pix_X,cls.pix_Y)
    
    @staticmethod
    def to_table(cls,version = 'test'):
        """ convert this to an `astropy.table.Table` """
        # currently the neighbor list is not supported, since
        # var-length arrays are not supported by astropy.table.Table
        return Table([cls.pix_id, cls.pix_X, cls.pix_Y, cls.pix_area,
                      cls.pix_neighbors],
                      names=['PixID', 'PixX', 'PixY', 'PixA', 'PixNeig'],
                      meta=dict(VERSION=version))

                    
npix_to_type = {(2048, 0.006): ('SST', 'GATE', 'rectangular'),
                 (2048, 0.042): ('LST', 'HESSII', 'hexagonal'),
                 (1141, None): ('MST', 'NectarCam', 'hexagonal'),
                 (1855, None): ('LST', 'LSTCam', 'hexagonal'),
                 (11328, None): ('SCT', 'SCTCam', 'rectangular')}
                 
def find_neighbor_pixels(pix_x, pix_y, rad):
    """uses a KD-Tree to quickly find nearest neighbors of the pixels in a
    camera. This function can be used to find the neighbor pixels if
    such a list is not already present in the file.

    Parameters
    ----------
    pix_x : array_like
        x position of each pixel
    pix_y : array_like
        y position of each pixels
    rad : float
        radius to consider neighbor it should be slightly larger
        than the pixel diameter.

    Returns
    -------
    array of neighbor indices in a list for each pixel
    """

    points = np.array([pix_x, pix_y]).T
    indices = np.arange(len(pix_x))
    kdtree = KDTree(points)
    neighbors = [kdtree.query_ball_point(p, r=rad) for p in points]
    for nn, ii in zip(neighbors, indices):
        nn.remove(ii)  # get rid of the pixel itself
    return neighbors
                      
@u.quantity_input
def guess_camera_geometry(pix_x: u.m, pix_y: u.m):
    """
    returns a the camera class, the pixel area, the pixel type and the
    distance between two pixels just from the pixel positions        
    
    Assumes:
    --------
    - the pixels are square or hexagonal
    - the first two pixels are adjacent
    
    Parameters
    ----------
    pix_x: array with units
        positions of pixels (x-coordinate)
    pix_y: array with units
        positions of pixels (y-coordinate
    
    Returns
    -------
    camera class, pixel area, pixel type, and distance between 2 pixels
    projected on the x-axis
    """

    try: cam_class,pix_type = guess_camera_type(len(pix_x))
    except:
        pix_type = -1
        
    dx = pix_x[1] - pix_x[0]
    dy = pix_y[1] - pix_y[0]
    dist = np.sqrt(dx ** 2 + dy ** 2)  # dist between two pixels
    tel_type, cam_id, pix_type = guess_camera_type(len(pix_x), \
    u.Quantity(dist,"m").value)

    if pix_type.startswith('hex'):
        rad = dist / np.sqrt(3)  # radius to vertex of hexagon
        area = rad ** 2 * (3 * np.sqrt(3) / 2.0)  # area of hexagon
    elif pix_type.startswith('rect'):
        area = dist ** 2
    else: pass

    pix_area=np.ones(pix_x.shape) * area

    return Camera(pix_id=np.arange(len(pix_x)),pix_X=pix_x,pix_Y=pix_y,
                          pix_area=np.ones(pix_x.shape) * pix_area,
                          pix_neighbors=find_neighbor_pixels(pix_x.value,
                                                         pix_y.value,
                                                         dx.value + 0.01),
                          pix_type=pix_type)

def guess_camera_type(npix,pix_sep):
    # dictionary to convert number of pixels to camera type for use in
    # guess_camera_geometry.
    # Key = (npix, pix_seperation_m)
    # Value = (type, subtype, pixtype)
    """
    guesses and returns the camera type using the number of pixels

    Parameters
    ----------
    npix: int
        number of pixels of the camera
    """
    global npix_to_type
    try:
        return npix_to_type[(npix, None)]
    except KeyError:
        return npix_to_type.get((npix,np.round(pix_sep,3)), \
        ('unknown', 'unknown', 'hexagonal'))

def make_rectangular_camera_geometry(npix_x=40, npix_y=40,
                                     range_x=(-0.5, 0.5), range_y=(-0.5, 0.5)):
    """Generate a simple camera with 2D rectangular geometry.
    Used for testing.
    Parameters
    ----------
    npix_x : int
        number of pixels in X-dimension
    npix_y : int
        number of pixels in Y-dimension
    range_x : (float,float)
        min and max of x pixel coordinates in meters
    range_y : (float,float)
        min and max of y pixel coordinates in meters
    Returns
    -------
    pix_id: int array
        Pixel IDs
    pix_x,pix_y: float arrays
        x and y positions of the pixels in the camera
    pix_area: float array
        pixel areas
    neighbors: list with next neighbors
    """
    bx = np.linspace(range_x[0], range_x[1], npix_x)
    by = np.linspace(range_y[0], range_y[1], npix_y)
    xx, yy = np.meshgrid(bx, by)
    pix_x = xx.ravel() * u.m
    pix_y = yy.ravel() * u.m

    pix_ids = np.arange(npix_x * npix_y)
    rr = np.ones_like(pix_x).value * (pix_x[1] - pix_x[0]) / 2.0
    pix_area = (2 * rr) ** 2
    neighbors = find_neighbor_pixels(pix_x.value, pix_y.value,
                                     rad=(rr.mean() * 2.001).value)
                              
    return Camera(pix_id = pix_ids,pix_X = pix_x, pix_Y = pix_y,
                  pix_area = pix_area,pix_neighbors = neighbors,
                  pix_type = 'rectangular')
                      
def rotate_camera(angle,pix_x: u.m,pix_y:u.m):
    """rotate the camera coordinates about the center of the camera by
    specified angle. Modifies the CameraGeometry in-place (so
    after this is called, the pix_x and pix_y arrays are
    rotated.
    Note:
    -----
    This is intended only to correct simulated data that are
    rotated by a fixed angle.  For the more general case of
    correction for camera pointing errors (rotations,
    translations, skews, etc), you should use a true coordinate
    transformation defined in `ctapipe.coordinates`.
    Parameters
    ----------
    angle: value convertable to an `astropy.coordinates.Angle`
        rotation angle with unit (e.g. 12 * u.deg), or "12d"
    pix_x: array with x-positions of the camera pixels
    pix_y: array with y-positions of the camera pixels
    """
    if type(pix_x) == astropy.table.column.Column:
        pix_x = pix_x*pix_x.unit
        pix_y = pix_y*pix_y.unit
    rotmat = rotation_matrix_2d(angle)
    rotated = np.dot(rotmat.T, [pix_x.value, pix_y.value])
    pix_x = rotated[0] * pix_x.unit
    pix_y = rotated[1] * pix_x.unit
    
    return pix_x, pix_y