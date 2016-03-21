# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utilities for reading or working with Camera geometry files
"""
import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import Angle
from scipy.spatial import cKDTree as KDTree

from .files import get_file_type
from ..utils.datasets import get_path
from ..utils.linalg import rotation_matrix_2d

__all__ = ['CameraGeometry',
           'make_rectangular_camera_geometry']


# dictionary to convert number of pixels to camera type for use in
# guess_camera_geometry.
# Key = (npix, pix_separation_m)
# Value = (type, subtype, pixtype, pixrotation, camrotation)
_npix_to_type = {
    (2048, 2.3):   ('SST', 'GATE', 'rectangular', 0 * u.degree, 0 * u.degree),
    (2048, 36.0):  ('LST', 'HESSII', 'hexagonal', 0 * u.degree, 0 * u.degree),
    (1855, 16.0):  ('MST', 'NectarCam', 'hexagonal', 0 * u.degree, -100.893 * u.degree),
    (1855, 28.0):  ('LST', 'LSTCam', 'hexagonal', 0. * u.degree, -100.893 * u.degree),
    (1296, None):  ('SST', 'SST-1m', 'hexagonal', 30 * u.degree, 0 * u.degree),
    (1764, None):  ('MST', 'FlashCam', 'hexagonal', 30 * u.degree, 0 * u.degree),
    (2368, None):  ('SST', 'ASTRI', 'rectangular', 0 * u.degree, 0 * u.degree),
    (11328, None): ('SCT', 'SCTCam', 'rectangular', 0 * u.degree, 0 * u.degree),
}


class CameraGeometry:
    """`CameraGeometry` is a class that stores information about a
    Cherenkov Camera that us useful for imaging algorithms and
    displays. It contains lists of pixel positions, areas, pixel
    shapes, as well as a neighbor (adjacency) list for each pixel.

    The class is intended to be generic, and work with any Cherenkov
    Camera geometry, including those that have square vs hexagonal
    pixels, gaps between pixels, etc.
    """

    def __init__(self, cam_id, pix_id, pix_x, pix_y,
                 pix_area, neighbors, pix_type, pix_rotation=0 * u.degree, cam_rotation=0 * u.degree):
        """
        Parameters
        ----------
        self: type
            description
        cam_id: camera id name or number
            camera identification string
        pix_id: array(int)
            pixels id numbers
        pix_x: array with units
            position of each pixel (x-coordinate)
        pix_y: array with units
            position of each pixel (y-coordinate)
        pix_area: array(float)
            surface area of each pixe
        neighbors: list(arrays)
            adjacency list for each pixel
        pix_type: string
            either 'rectangular' or 'hexagonal'
        pix_rotation: value convertable to an `astropy.coordinates.Angle`
            rotation angle with unit (e.g. 12 * u.deg), or "12d"
        cam_rotation: overall camera rotation with units

        """
        self.cam_id = cam_id
        self.pix_id = pix_id
        self.pix_x = pix_x
        self.pix_y = pix_y
        self.pix_area = pix_area
        self.neighbors = neighbors
        self.pix_type = pix_type
        self.pix_rotation = Angle(pix_rotation)
        self.rotate(cam_rotation)


    @classmethod
    def guess(cls, pix_x, pix_y, optical_foclen):
        """
        Construct a `CameraGeometry` by guessing the appropriate quantities
        from a list of pixel positions.
        """
        return guess_camera_geometry(pix_x, pix_y, optical_foclen)

    @classmethod
    def from_name(cls, name, tel_id):
        """
        Construct a `CameraGeometry` from the name of the instrument and
        telescope id, if it can be found in a standard database.
        """
        return get_camera_geometry(name, tel_id)

    @classmethod
    def from_file(cls, filename, tel_id):
        """
        Construct a `CameraGeometry` from the a data file
        """
        filetype = get_file_type(filename)
        if filetype == 'simtel':
            return _load_camera_geometry_from_hessio_file(tel_id, filename)
        else:
            raise TypeError("File type {} not supported".format(filetype))

    def to_table(self):
        """ convert this to an `astropy.table.Table` """
        # currently the neighbor list is not supported, since
        # var-length arrays are not supported by astropy.table.Table
        return Table([self.pix_id, self.pix_x, self.pix_y, self.pix_area],
                     names=['pix_id', 'pix_x', 'pix_y', 'pix_area'],
                     meta=dict(pix_type=self.pix_type,
                               TYPE='CameraGeometry',
                               CAM_ID=self.cam_id))

    def rotate(self, angle):
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

        """
        rotmat = rotation_matrix_2d(angle)
        rotated = np.dot(rotmat.T, [self.pix_x.value, self.pix_y.value])
        self.pix_x = rotated[0] * self.pix_x.unit
        self.pix_y = rotated[1] * self.pix_x.unit
        self.pix_rotation -= angle


# ======================================================================
# utility functions:
# ======================================================================


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

    points = np.array([pix_x, pix_y]).T
    indices = np.arange(len(pix_x))
    kdtree = KDTree(points)
    neighbors = [kdtree.query_ball_point(p, r=rad) for p in points]
    for nn, ii in zip(neighbors, indices):
        nn.remove(ii)  # get rid of the pixel itself
    return neighbors


def _guess_camera_type(npix, optical_foclen):
    global _npix_to_type

    try:
        return _npix_to_type[(npix, None)]
    except KeyError:
        return _npix_to_type.get((npix, round(optical_foclen.value, 1)),
                                 ('unknown', 'unknown', 'hexagonal', 0 * u.degree, 0 * u.degree))


@u.quantity_input
def guess_camera_geometry(pix_x: u.m, pix_y: u.m, optical_foclen: u.m):
    """ returns a CameraGeometry filled in from just the x,y positions

    Assumes:
    --------
    - the pixels are square or hexagonal
    """

#    dx = pix_x[1] - pix_x[0]    <=== Not adjacent for DC-SSTs!!
#    dy = pix_y[1] - pix_y[0]

    pixsep = []
    for ipix in range(1, len(pix_x)):
        dx = pix_x[ipix] - pix_x[0]
        dy = pix_y[ipix] - pix_y[0]        
        pixsep.append (np.sqrt(dx ** 2 + dy ** 2))  # dist between pixels 0 and ipix
        
    dist = min(pixsep)
        
    tel_type, cam_id, pix_type, pix_rotation, cam_rotation = _guess_camera_type(
        len(pix_x), optical_foclen
    )

    if pix_type.startswith('hex'):
        rad = dist / np.sqrt(3)  # radius to vertex of hexagon
        area = rad ** 2 * (3 * np.sqrt(3) / 2.0)  # area of hexagon
    elif pix_type.startswith('rect'):
        area = dist ** 2
    else:
        raise KeyError("unsupported pixel type")

    return CameraGeometry(
        cam_id=cam_id,
        pix_id=np.arange(len(pix_x)),
        pix_x=pix_x,
        pix_y=pix_y,
        pix_area=np.ones(pix_x.shape) * area,
        neighbors=find_neighbor_pixels(
            pix_x.value,
            pix_y.value,
            1.4 * dist.value,
        ),
        pix_type=pix_type,
        pix_rotation=pix_rotation,
        cam_rotation=cam_rotation,
    )


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

    geom = _load_camera_table_from_file(cam_id, geomfile=geomfile)
    neigh_list = geom['PIX_NEIG'].data
    neigh = np.ma.masked_array(neigh_list, neigh_list < 0),

    # put them all in units of M (conversions are automatic)
    xx = u.Quantity(geom['PIX_POSX'], u.m)
    yy = u.Quantity(geom['PIX_POSY'], u.m)
    dd = u.Quantity(geom['PIX_DIAM'], u.m)
    aa = u.Quantity(geom['PIX_AREA'], u.m ** 2)

    if recalc_neighbors is True:
        neigh = find_neighbor_pixels(xx.value, yy.value,
                                     (dd.mean() + 0.01 * u.m).value)

    return CameraGeometry(
        cam_id=cam_id,
        pix_id=np.array(geom['PIX_ID']),
        pix_x=xx,
        pix_y=yy,
        pix_area=aa,
        neighbors=neigh,
        pix_type='hexagonal'
    )


def _load_camera_table_from_file(cam_id, geomfile='chercam.fits.gz'):
    filetype = get_file_type(geomfile)
    if filetype == 'fits':
        return _load_camera_geometry_from_fits_file(cam_id, geomfile)
    else:
        raise NameError("file type not supported")


def _load_camera_geometry_from_fits_file(cam_id, geomfile='chercam.fits.gz'):
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


def _load_camera_geometry_from_hessio_file(tel_id, filename):
    import hessio  # warning, non-rentrant!
    hessio.file_open(filename)
    events = hessio.move_to_next_event()
    next(events)  # load at least one event to get all the headers
    pix_x, pix_y = hessio.get_pixel_position(tel_id)
    optical_foclen = hessio.get_optical_foclen(tel_id)

    hessio.close_file()
    return CameraGeometry.guess(pix_x * u.m, pix_y * u.m, optical_foclen)


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
