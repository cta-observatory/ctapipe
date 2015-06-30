"""Utilities for reading or working with Camera geometry files

TODO:
-----

 - don't use `namedtuple` for CameraGeometry, since it's immutable and thus is
   pass-by-value (which could be slow). 

"""
import numpy as np
from astropy.table import Table
from astropy import units as u
from collections import namedtuple
from ctapipe.utils.datasets import get_path

__all__ = ['CameraGeometry',
           'get_camera_geometry',
           'load_camera_geometry_from_file',
           'make_rectangular_camera_geometry'
           ]

#__doctest_skip__ = ['load_camera_geometry_from_file'  ]
CameraGeometry = namedtuple("CameraGeometry",
                            ['cam_id', 'pix_id',
                             'pix_x', 'pix_y', 'pix_r',
                             'pix_area',
                             'neighbor_ids',
                             'pix_type'])

def get_camera_geometry(instrument_name, cam_id):
    """Helper function to provide the camera geometry definition for a
    camera by name

    Parameters
    ----------
    instrument_name: ['hess',]
        name of instrument
    cam_id: int
        identifier of camera, in case of multiple versions

    Returns
    -------
    a `CameraGeometry` object


    Example
    -------

    >>> geom_ct1 = get_camera_geometry( "hess", 1 )
    >>> neighbors_pix_1 = geom_ct1.pix_id[geom_ct1.neighbor_ids[1].compressed()]

    """

    # let's assume the instrument name is encoded in the
    # filename
    name = instrument_name.lower()
    geomfile = get_path('{}_camgeom.fits.gz'.format(name))

    geom = load_camera_geometry_from_file(cam_id, geomfile=geomfile)
    neigh = geom['PIX_NEIG'].data
    
    return CameraGeometry(
        cam_id=cam_id,
        pix_id=geom['PIX_ID'].data,
        pix_x=geom['PIX_POSX'].data * geom['PIX_POSX'].unit,
        pix_y=geom['PIX_POSY'].data * geom['PIX_POSY'].unit,
        pix_r=geom['PIX_DIAM'] / 2.0,
        pix_area=geom['PIX_AREA'],
        neighbor_ids=np.ma.masked_array(neigh, neigh < 0),
        pix_type='hexagonal')

def load_camera_geometry_from_file(cam_id, geomfile='chercam.fits.gz'):
    """
    Read camera geometry from a  FITS file with a CHERCAM extension.

    Parameters
    ----------

    cam_id : int
        ID number of camera in the fits file
    geomfile : str
        FITS file containing camera geometry in CHERCAM extension

    Returns
    -------

    a `CameraGeometry` object

    """
    camtable = Table.read(geomfile, hdu="CHERCAM")
    geom = camtable[camtable['CAM_ID'] == cam_id]
    return geom


def make_rectangular_camera_geometry(npix_x=40, npix_y=40,
                                     range_x=(-0.5, 0.5), range_y=(-0.5, 0.5)):
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
    xx, yy = np.meshgrid(bx, by)
    xx = xx.ravel()
    yy = yy.ravel()
    
    ids = np.arange(npix_x * npix_y)
    
    neighs = None  # todo

    return CameraGeometry(
        cam_id=-1,
        pix_id=ids,
        pix_x=xx * u.m,
        pix_y=yy * u.m,
        pix_r=np.ones_like(xx) * (xx[1]-xx[0])/2.0 * u.m,
        pix_area=np.ones_like(xx)* (xx[1]-xx[0])*(yy[1]-yy[0]) * u.m**2,
        neighbor_ids=None,
        pix_type='rectangular')
    
