# Licensed under a 3-clause BSD style license - see LICENSE.rst
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
from scipy.spatial import cKDTree as KDTree

__all__ = ['CameraGeometry',
           'get_camera_geometry',
           'load_camera_geometry_from_file',
           'make_rectangular_camera_geometry',
           'find_neighbor_pixels', 'guess_camera_geometry',
           ]


# dictionary to convert number of pixels to camera type for use in
# guess_camera_geometry
_npix_to_type = {2048: ('SST', 'rectangular'),
                 1141: ('MST', 'hexagonal'),
                 1855: ('LST', 'hexagonal'),
                 11328: ('SST', 'rectangular')}


#__doctest_skip__ = ['load_camera_geometry_from_file'  ]
CameraGeometry = namedtuple("CameraGeometry",
                            ['cam_id', 'pix_id',
                             'pix_x', 'pix_y',
                             'pix_area',
                             'neighbors',
                             'pix_type'])
"""Camera geometry.

TODO: describe a bit what this is ...
"""


def find_neighbor_pixels(pix_x, pix_y, rad):
    """use a KD-Tree to quickly find nearest neighbors of the pixels in a
    camera. This function can be used to find the neighbor pixels if
    they are not already present in a camera geometry file.

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
    
    points = np.array([pix_x,pix_y]).T
    indices = np.arange(len(pix_x))
    kdtree = KDTree(points)
    neighbors = [kdtree.query_ball_point(p, r=rad) for p in points]
    for nn, ii in zip(neighbors, indices):
        nn.remove(ii)  # get rid of the pixel itself
    return neighbors


def guess_camera_type(npix):
    global _npix_to_type
    return _npix_to_type.get(npix, ('unknown', 'hexagonal'))


@u.quantity_input
def guess_camera_geometry(pix_x: u.m, pix_y: u.m):
    """ returns a CameraGeometry filled in from just the x,y positions """

    rad = 0.5 * \
        np.sqrt((pix_x[1] - pix_x[0]) ** 2 + (pix_y[1] - pix_y[0]) ** 2)

    cam_id, pix_type = guess_camera_type(len(pix_x))

    return CameraGeometry(cam_id=cam_id,
                          pix_id=np.arange(len(pix_x)),
                          pix_x=pix_x,
                          pix_y=pix_y,
                          pix_area=np.pi * np.ones_like(pix_x) * rad ** 2,
                          neighbors=find_neighbor_pixels(pix_x.value, pix_y.value,
                                                         rad.value + 0.01),
                          pix_type=pix_type)


def get_camera_geometry(instrument_name, cam_id, recalc_neighbors=True):
    """Helper function to provide the camera geometry definition for a
    camera by name.

    Parameters
    ----------
    instrument_name : {'hess'}
        name of instrument
    cam_id : int
        identifier of camera, in case of multiple versions
    recalc_neighbors : bool
        if True, recalculate the neighbor pixel list, otherwise
        use what is in the file

    Returns
    -------
    a `CameraGeometry` object

    Examples
    --------

    >>> geom_ct1 = get_camera_geometry( "hess", 1 )
    >>> neighbors_pix_1 = geom_ct1.pix_id[geom_ct1.neighbors[1]]
    """

    # let's assume the instrument name is encoded in the
    # filename
    name = instrument_name.lower()
    geomfile = get_path('{}_camgeom.fits.gz'.format(name))

    geom = load_camera_geometry_from_file(cam_id, geomfile=geomfile)
    neigh_list = geom['PIX_NEIG'].data
    neigh = np.ma.masked_array(neigh_list, neigh_list < 0),

    # put them all in units of M (conversions are automatic)
    xx = u.Quantity(geom['PIX_POSX'], u.m)
    yy = u.Quantity(geom['PIX_POSY'], u.m)
    dd = u.Quantity(geom['PIX_DIAM'], u.m)
    aa = u.Quantity(geom['PIX_AREA'], u.m**2)

    if recalc_neighbors is True:
        neigh = find_neighbor_pixels(xx.value, yy.value,
                                     (dd.mean() + 0.01*u.m).value)

    return CameraGeometry(
        cam_id=cam_id,
        pix_id=np.array(geom['PIX_ID']),
        pix_x=xx,
        pix_y=yy,
        pix_area=aa,
        neighbors=neigh,
        pix_type='hexagonal'
    )


def load_camera_geometry_from_file(cam_id, geomfile='chercam.fits.gz'):
    """
    Read camera geometry from a  FITS file with a ``CHERCAM`` extension.

    Parameters
    ----------

    cam_id : int
        ID number of camera in the fits file
    geomfile : str
        FITS file containing camera geometry in ``CHERCAM`` extension

    Returns
    -------

    a `CameraGeometry` object

    """
    camtable = Table.read(geomfile, hdu="CHERCAM")
    geom = camtable[camtable['CAM_ID'] == cam_id]
    return geom


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
    xx = xx.ravel() * u.m
    yy = yy.ravel() * u.m

    ids = np.arange(npix_x * npix_y)
    rr = np.ones_like(xx).value * (xx[1] - xx[0]) / 2.0
    nn = find_neighbor_pixels(xx.value, yy.value,
                              rad=(rr.mean() * 2.001).value)
    return CameraGeometry(
        cam_id=-1,
        pix_id=ids,
        pix_x=xx * u.m,
        pix_y=yy * u.m,
        pix_area=(2 * rr) ** 2,
        neighbors=nn,
        pix_type='rectangular')
