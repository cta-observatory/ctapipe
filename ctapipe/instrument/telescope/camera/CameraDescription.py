import pyhessio as h
from astropy import units as u
from astropy.table import Table
import numpy as np
import os
import textwrap
from scipy.spatial import cKDTree as KDTree

from ctapipe.utils.linalg import rotation_matrix_2d
from ctapipe.instrument.util_functions import get_file_type

__all__ = ['initialize','rotate','write_table']

class Initialize:

    """`Initialize` is a class containing the initialize functions for
    the different file extensions"""
    
    @u.quantity_input
    def _initialize_hessio(filename,tel_id,item):
        """
        reads the Camera data out of the open hessio file
        
        Parameters
        ----------
        filename: string
            name of the hessio file (must be a hessio file!)
        tel_id: int
            ID of the telescope whose optics information should be loaded
        """
        cam_fov = -1*u.degree
        pix_posX = h.get_pixel_position(tel_id)[0]*u.m
        pix_posY = h.get_pixel_position(tel_id)[1]*u.m
        pix_posZ = [-1]*u.m
        pix_id = np.arange(len(pix_posX))
        
        cam_class,pix_area,pix_type,dx = _guess_camera_geometry(pix_posX,
                                                                pix_posY)
        
        try: cam_class = h.get_camera_class(tel_id)
        except: pass
        
        try: pix_area = h.get_pixel_area(tel_id)
        except: pass
        
        try: pix_type = h.get_pixel_type(tel_id)
        except: pass
        
        try: pix_neighbors = h.get_pix_neighbors(tel_id)
        except:
            pix_neighbors = _find_neighbor_pixels(pix_posX.value,
                                                  pix_posY.value,
                                                  dx.value + 0.01)
        fadc_pulsshape = [[-1],[-1]]
        #to use this, one has to go through every event of the run...
        #n_channel = h.get_num_channel(tel_id)
        #ld.channel_num = n_channel
        #for chan in range(n_channel):
        #    ld.adc_samples.append(h.get_adc_sample(tel_id,chan).tolist())
    
        return (cam_class,cam_fov,pix_id,pix_posX,pix_posY,pix_posZ,pix_area,
                pix_type,pix_neighbors,fadc_pulsshape)
            
    def _initialize_fits(filename,tel_id,item):
        """
        reads the Camera data out of the open fits file
        
        Parameters
        ----------
        filename: string
            name of the fits file (must be a fits file!)
        tel_id: int
            ID of the telescope or of the camera whose optics information
            should be loaded
            
        item: HDUList
            HDUList of the fits file
        """
        hdulist = item
        
        cam_class = -1
        cam_fov = -1*u.degree
        lid = -1
        pix_posX = [-1]*u.m
        pix_posY = [-1]*u.m
        pix_posZ = [-1]*u.m
        pix_area = [-1]*u.m**2
        pix_type = -1
        pix_neighbors = [-1]
        fadc_pulsshape = [[-1],[-1]]
        
        
        for i in range(len(hdulist)):
            teles = hdulist[i].data
            
            try: cam_class = teles['CamClass'][teles['TelID']==tel_id][0]
            except: pass
        
            try: cam_fov = teles['FOV'][teles['TelID']==tel_id][0]*u.degree
            except: pass
        
            try: lid = teles['L0ID'][teles['TelID']==tel_id][0]
            except: pass
        
            try: pix_id = teles['PixelID'][teles['L0ID']==lid]
            except: pass
        
            try: pix_posX = teles['Pix_PosX'][teles['L0ID']==lid]*u.m
            except: pass
        
            try: pix_posY = teles['Pix_PosY'][teles['L0ID']==lid]*u.m
            except: pass
        
            try: pix_posZ = teles['Pix_PosZ'][teles['L0ID']==lid]*u.m
            except: pass
    
            try: pix_area = teles['Pix_Area'][teles['L0ID']==lid]*u.m
            except: pass
        
            try: pix_type = teles['Pix_Type'][teles['L0ID']==lid]*u.m**2
            except: pass
        
            try: pix_neighbors = teles['Pix_Neighbors'][teles['L0ID']==lid]
            except: pass
        
            try: fadc_pulsshape = teles['FADC_PulseShape'][teles['L0ID']==lid]
            except: pass
    
        return (cam_class,cam_fov,pix_id,pix_posX,pix_posY,pix_posZ,pix_area,
                pix_type,pix_neighbors,fadc_pulsshape)
    
    def _initialize_ascii(filename,tel_id,item):
        """
        reads the Camera data out of the ASCII file
        
        Parameters
        ----------
        filename: string
            name of the ASCII file (must be an ASCII config file!)
        tel_id: int
            ID of the telescope whose optics information should be loaded (must
            not be given)
        item: python module
            python module created from an ASCII file using imp.load_source
        """
        dirname = os.path.dirname(filename)        
        
        try: cam_class = item.cam_class[0]
        except: cam_class = -1
        
        try: cam_fov = item.cam_fov[0]*u.degree
        except: cam_fov = -1*u.degree
        
        try: pix_id = item.pix_id
        except: pix_id = [-1]
        
        try: pix_posX = item.pix_posX*u.m
        except: pix_posX = [-1]*u.m
        
        try: pix_posY = item.pix_posY*u.m
        except: pix_posY = [-1]*u.m
        
        try: pix_posZ = item.pix_posZ*u.m
        except: pix_posZ = [-1]*u.m
                
        try: pix_area = item.pix_area*u.m**2
        except: pix_area = [-1]*u.m**2
        
        try: pix_type = item.pix_type[0]
        except: pix_type = [-1]
        
        try: pix_neighbors = item.pix_neighbors
        except: pix_neighbors = [-1]
        
        try:
            time, pulse_shape = np.loadtxt(dirname+'/'+textwrap.dedent(item.fadc_pulse_shape[0]),
                                           unpack=True)
            fadc_pulsshape = [time,pulse_shape]
        except: fadc_pulsshape = [[-1],[-1]]
        
        
        return (cam_class,cam_fov,pix_id,pix_posX,pix_posY,pix_posZ,pix_area,
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


def initialize(filename,tel_id,item):
    """
    calls the specific initialize function depending on the file
    extension of the given file. The file must already be open/have
    been loaded. The return value of the opening/loading process
    must be given as an argument (item).
    
    Parameters
    ----------
    filename: string
        name of the file
    tel_id: int
        ID of the telescope whose optics information should be loaded
    item: of various type depending on the file extension
        return value of the opening/loading process of the file
    """
    ext = get_file_type(filename)

    function = getattr(Initialize,"_initialize_%s" % ext)
    return function(filename,tel_id,item)       
    
    #if 'simtel.gz' in filename:
    #    return _initialize_hessio(filename,tel_id)
    #elif 'fits' in filename:
    #    return _initialize_fits(filename,tel_id,item)
    
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





