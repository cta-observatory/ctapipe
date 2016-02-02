from astropy import units as u
from astropy.table import Table
import numpy as np
from scipy.spatial import cKDTree as KDTree
from ctapipe.utils.linalg import rotation_matrix_2d

__all__ = ['get_data','rotate','write_table']
    
@u.quantity_input
def get_data(instr_table,tel_id):
    """
    reads the Camera data out of the instrument table
    
    Parameters
    ----------
    instr_table: astropy table
        name of the astropy table where the whole instrument data read from
        the file is stored
    tel_id: int
        ID of the telescope whose optics information should be loaded
    """
    cam_class = -1
    cam_fov = -1*u.degree
    pix_id = [-1]
    pix_posX = [-1]*u.m
    pix_posY = [-1]*u.m
    pix_area = [-1]*u.m**2
    pix_type = -1
    pix_neighbors =[-1]
    fadc_pulsshape = [[-1],[-1]]
        
    
    #tel_table,cam_table,opt_table = instr_table    
    
    for i in range(len(instr_table)):
        try: tel_id_bool = instr_table[i]['TelID']==tel_id
        except: pass
    
    for i in range(len(instr_table)):
        try:
            cam_group = instr_table[i].group_by('TelID')
            mask = (cam_group.groups.keys['TelID'] == tel_id)
        except: pass
    
        try:
            pix_posX = cam_group.groups[mask]['PixX']
            if pix_posX.unit == None:
                pix_posX.unit = u.mm
        except: pass
        
        try:
            pix_posY = cam_group.groups[mask]['PixY']
            if pix_posY.unit == None:
                pix_posY.unit = u.mm
        except: pass
        
        try: pix_id = cam_group.groups[mask]['PixelID']
        except: pass
        
        try: cam_fov = instr_table[i][tel_id_bool]['FOV']
        except TypeError:
            cam_fov = instr_table[i][tel_id_bool]['FOV']*u.degree
        except: pass
    
    try:
        cam_class_prime,pix_area_prime,pix_type_prime,dx = _guess_camera_geometry(pix_posX,
                                                            pix_posY)
        pix_area = pix_area*(pix_posX.unit)**2
        dx = dx*pix_posX.unit
    except: pass
    
    for i in range(len(instr_table)):
    
        try: cam_class = instr_table[i][tel_id_bool]['CameraClass']
        except:
            try: cam_class = cam_class_prime
            except: pass
        
        try:
            pix_area = cam_group.groups[mask]['PixArea']
            if pix_area.unit == None:
                pix_area.unit = u.mm**2
        except:
            try: pix_area = pix_area_prime
            except: pass
        
        try: pix_type = instr_table[i][tel_id_bool]['PixType']
        except:
            try: pix_type = pix_type_prime
            except: pass
        
        try: pix_neighbors = cam_group.groups[mask]['PixNeighbors']
        except:
            try:
                pix_neighbors = _find_neighbor_pixels(pix_posX,pix_posY,
                                                  dx.value + 0.01)
            except: pass

    return (cam_class,cam_fov,pix_id,pix_posX,pix_posY,pix_area,
            pix_type,pix_neighbors,fadc_pulsshape)

# dictionary to convert number of pixels to camera type for use in
# guess_camera_geometry
_npix_to_type = {2048: ('SST', 'rectangular'),
                 1141: ('MST', 'hexagonal'),
                 1855: ('LST', 'hexagonal'),
                 11328: ('SST', 'rectangular')}

def _guess_camera_type(npix):
    """
    guesses and returns the camera type using the number of pixels

    Parameters
    ----------
    npix: int
        number of pixels of the camera
    """
    global _npix_to_type
    return _npix_to_type.get(npix)

@u.quantity_input
def _guess_camera_geometry(pix_x: u.m, pix_y: u.m):
    """ returns a CameraGeometry filled in from just the x,y positions
    
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

    try: cam_class,pix_type = _guess_camera_type(len(pix_x))
    except:
        cam_class = -1
        pix_type = -1
        
    dx = pix_x[1] - pix_x[0]
    dy = pix_y[1] - pix_y[0]
    dist = np.sqrt(dx ** 2 + dy ** 2)  # dist between two pixels

    if pix_type.startswith('hex'):
        rad = dist / np.sqrt(3)  # radius to vertex of hexagon
        area = rad ** 2 * (3 * np.sqrt(3) / 2.0)  # area of hexagon
    elif pix_type.startswith('rect'):
        area = dist ** 2
    else:
        area = -1 #unsupported pixel type
        pix_area = [-1]*u.m**2
    
    if area != -1:
        pix_area = np.ones(pix_x.shape) * area

    return cam_class,pix_area,pix_type,dx

def _find_neighbor_pixels(pix_x, pix_y, rad):
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

    
def write_table(tel_id,cam_class,pix_id,pix_x,pix_y,pix_area,pix_type):
    """
    writes values into an `astropy.table.Table`
    
    Parameters
    ----------
    tel_id: int
        ID of the telescope
    cam_class: string
        camera class
    pix_id: int array
        IDs of pixels of a given telescope
    pix_x:astropy.units array
        x-positions of the pixels
    pix_y:astropy.units array
        y-positions of the pixel
    pix_area: astropy.units array
        areas of the pixe
    """
    # currently the neighbor list is not supported, since
    # var-length arrays are not supported by astropy.table.Table
    return Table([pix_id,pix_x,pix_y,pix_area],
                 names=['pix_id', 'pix_x', 'pix_y', 'pix_area'],
                 meta=dict(pix_type=pix_type,
                           TYPE='CameraGeometry',
                           TEL_ID=tel_id,
                           CAM_CLASS=cam_class))

def rotate(pix_x,pix_y,angle):
    """rotate the camera coordinates about the center of the camera by
    specified angle.
    
    Parameters
    ----------
    pix_x: x-position of pixel with unit
    pix_y: y-position of pixel with unit
    angle: value convertable to an `astropy.coordinates.Angle`
        rotation angle with unit (e.g. 12 * u.deg), or "12d"
        
    Returns
    -------
    roated x- and y-positions
    """
    rotmat = rotation_matrix_2d(angle)
    rotated = np.dot(rotmat.T, [pix_x.value,pix_y.value])
    pix_x = rotated[0] * pix_x.unit
    pix_y = rotated[1] * pix_x.unit
    
    return pix_x,pix_y
    
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
    CameraGeometry object
    """
    bx = np.linspace(range_x[0], range_x[1], npix_x)
    by = np.linspace(range_y[0], range_y[1], npix_y)
    xx, yy = np.meshgrid(bx, by)
    pix_x = xx.ravel() * u.m
    pix_y = yy.ravel() * u.m

    pix_id = np.arange(npix_x * npix_y)
    rr = np.ones_like(xx).value * (xx[1] - xx[0]) / 2.0
    pix_area = (2 * rr) ** 2
    neighbors = _find_neighbor_pixels(xx.value, yy.value,
                              rad=(rr.mean() * 2.001).value)
    pix_type = 'rectangular'
    cam_id = -1
    return (cam_id,pix_id,pix_x*u.m,pix_y*u.m,pix_area,neighbors,pix_type)





