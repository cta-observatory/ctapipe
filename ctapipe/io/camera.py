"""
Utilities for reading or working with Camera geometry files
"""

import numpy as np
from astropy.table import Table
from astropy import units as u

__all__ = ['CameraGeometry',
           'load_camera_geometry',
           'make_rectangular_camera_geometry']

CameraGeometry = namedtuple("CameraGeometry",
                           "pix_id, pix_x, pix_y, neighbor_ids, pix_type")

def load_camera_geometry(cam_id, geomfile='chercam.fits.gz'):
    """
    Read camera geometry from a  FITS file with a CHERCAM extension

    Parameters
    ----------

    cam_id : int
        ID number of camera in the fits file
    geomfile: str
        FITS file containing camera geometry in CHERCAM extension
    
    Returns
    -------
    a :class:CameraGeometry object

    """
    camtable = Table.read(geomfile, extension="CHERCAM")
    geom = camtable[camtable['CAM_ID'] == cam_id]

    return CameraGeometry(
        pix_id=geom['PIX_ID'].data,
        pix_x=geom['PIX_POSX'].data * geom['PIX_POSX'].unit,
        pix_y=geom['PIX_POSY'].data * geom['PIX_POSY'].unit,
        neighbor_ids=geom['PIX_NEIG'].data,
        pix_type='hexagonal'  )


def make_rectangular_camera_geometry(npix_x=40, npix_y=40,
                                     range_x=(-0.5,0.5), range_y=(-0.5,0.5)):
    """generates a simple camera with 2D rectangular geometry, for
    testing purposes

    Parameters
    ----------
    npix_x: int
        number of pixels in X-dimension
    npix_y: int
        number of pixels in Y-dimension
    range_x: (float,float)
        min and max of x pixel coordinates in meters
    range_y: (float,float)
        min and max of y pixel coodinates in meters
    
    Returns
    -------
    CameraGeometry object

    """
    bx = np.linspace(range_x[0], range_x[1], npix_x)
    by = np.linspace(range_y[0], range_y[1], npix_y)
    xx,yy = np.meshgrid(bx,by)
    
    ids = np.arange(npix_x*npix_y)
    neighs = None # todo
    
    return CameraGeometry(ids,xx*u.m,yy*u.m,neighs, pix_type="rectangular")
    
