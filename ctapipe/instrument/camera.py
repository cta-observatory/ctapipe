# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utilities for reading or working with Camera geometry files
"""
import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from astropy.table import Table
from astropy.utils import lazyproperty
from ctapipe.io.files import get_file_type
from ctapipe.utils.datasets import get_path
from ctapipe.utils.linalg import rotation_matrix_2d
from scipy.spatial import cKDTree as KDTree

__all__ = ['CameraGeometry',
           'make_rectangular_camera_geometry']

# dictionary to convert number of pixels to camera + the focal length of the
# telescope into a camera type for use in `CameraGeometry.guess()`
#     Key = (num_pix, focal_length_in_meters)
#     Value = (type, subtype, pixtype, pixrotation, camrotation)
_CAMERA_GEOMETRY_TABLE = {
    (2048, 2.3): ('SST', 'GCT', 'rectangular', 0 * u.degree, 0 * u.degree),
    (2048, 2.2): ('SST', 'GCT', 'rectangular', 0 * u.degree, 0 * u.degree),
    (2048, 36.0): ('LST', 'HESSII', 'hexagonal', 0 * u.degree, 0 * u.degree),
    (1855, 16.0): (
        'MST', 'NectarCam', 'hexagonal', 0 * u.degree, -100.893 * u.degree),
    (1855, 28.0): (
        'LST', 'LSTCam', 'hexagonal', 0. * u.degree, -100.893 * u.degree),
    (1296, None): ('SST', 'SST-1m', 'hexagonal', 30 * u.degree, 0 * u.degree),
    (1764, None): ('MST', 'FlashCam', 'hexagonal', 30 * u.degree, 0 * u.degree),
    (2368, None): ('SST', 'ASTRI', 'rectangular', 0 * u.degree, 0 * u.degree),
    (11328, None): ('SCT', 'SCTCam', 'rectangular', 0 * u.degree, 0 * u.degree),
}


class CameraGeometry:
    """`CameraGeometry` is a class that stores information about a
    Cherenkov Camera that us useful for imaging algorithms and
    displays. It contains lists of pixel positions, areas, pixel
    shapes, as well as a neighbor (adjacency) list and matrix for each pixel. 
    In general the neighbor_matrix attribute should be used in any algorithm 
    needing pixel neighbors, since it is much faster. See for example 
    `ctapipe.image.tailcuts_clean` 

    The class is intended to be generic, and work with any Cherenkov
    Camera geometry, including those that have square vs hexagonal
    pixels, gaps between pixels, etc.
    
    You can construct a CameraGeometry either by specifying all data, 
    or using the `CameraGeometry.guess()` constructor, which takes metadata 
    like the pixel positions and telescope focal length to look up the rest 
    of the data. Note that this function is memoized, so calling it multiple 
    times with the same inputs will give back the same object (for speed).
    
    """

    _geometry_cache = {}  # dictionary CameraGeometry instances for speed

    def __init__(self, cam_id, pix_id, pix_x, pix_y, pix_area, pix_type,
                 pix_rotation=0 * u.degree, cam_rotation=0 * u.degree,
                 neighbors=None):
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
        self.pix_type = pix_type
        self.pix_rotation = Angle(pix_rotation)
        self.cam_rotation = Angle(cam_rotation)
        self._precalculated_neighbors = neighbors
        # FIXME the rotation does not work on 2D pixel grids
        if len(pix_x.shape) == 1:
            self.rotate(cam_rotation)
            self.cam_rotation = cam_rotation#Angle(0 * u.deg)

    def __eq__(self, other):
        return ( (self.cam_id == other.cam_id)
                 and (self.pix_x == other.pix_x).all()
                 and (self.pix_y == other.pix_y).all()
                 and (self.pix_type == other.pix_type)
                 and (self.pix_rotation == other.pix_rotation)
                 and (self.pix_type == other.pix_type)
                )

    @classmethod
    @u.quantity_input
    def guess(cls, pix_x: u.m, pix_y: u.m, optical_foclen: u.m):
        """ 
        Construct a `CameraGeometry` by guessing the appropriate quantities
        from a list of pixel positions and the focal length. 
        """
        # only construct a new one if it has never been constructed before,
        # to speed up access. Otherwise return the already constructed instance
        # the identifier uses the values of pix_x (which are converted to a
        # string to make them hashable) and the optical_foclen. So far,
        # that is enough to uniquely identify a geometry.
        identifier = (pix_x.value.tostring(), optical_foclen)
        if identifier in CameraGeometry._geometry_cache:
            return CameraGeometry._geometry_cache[identifier]

        # now try to determine the camera type using the map defined at the
        # top of this file.
        dist = _get_min_pixel_seperation(pix_x, pix_y)

        tel_type, cam_id, pix_type, pix_rotation, cam_rotation = \
            _guess_camera_type(len(pix_x), optical_foclen)

        if pix_type.startswith('hex'):
            rad = dist / np.sqrt(3)  # radius to vertex of hexagon
            area = rad ** 2 * (3 * np.sqrt(3) / 2.0)  # area of hexagon
        elif pix_type.startswith('rect'):
            area = dist ** 2
        else:
            raise KeyError("unsupported pixel type")

        instance = cls(
            cam_id=cam_id,
            pix_id=np.arange(len(pix_x)),
            pix_x=pix_x,
            pix_y=pix_y,
            pix_area=np.ones(pix_x.shape) * area,
            neighbors=None,
            pix_type=pix_type,
            pix_rotation=Angle(pix_rotation),
            cam_rotation=Angle(cam_rotation),
        )

        CameraGeometry._geometry_cache[identifier] = instance
        return instance

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
                     meta=dict(PIX_TYPE=self.pix_type,
                               TAB_TYPE='ctapipe.instrument.CameraGeometry',
                               TAB_VER='1.0',
                               CAM_ID=self.cam_id,
                               PIX_ROT=self.pix_rotation.deg,
                               CAM_ROT=self.cam_rotation.deg,
                ))


    @classmethod
    def from_table(cls, url_or_table, **kwargs):
        """
        Load a CameraGeometry from an `astropy.table.Table` instance or a 
        file that is readable by `astropy.table.Table.read()`
         
        Parameters
        ----------
        url_or_table: string or astropy.table.Table
            either input filename/url or a Table instance
        
        format: str
            astropy.table format string (e.g. 'ascii.ecsv') in case the 
            format cannot be determined from the file extension
            
        kwargs: extra keyword arguments
            extra arguments passed to `astropy.table.read()`, depending on 
            file type (e.g. format, hdu, path)


        """

        tab = url_or_table
        if not isinstance(url_or_table, Table):
            tab = Table.read(url_or_table, **kwargs)

        return cls(
            cam_id=tab.meta['CAM_ID'],
            pix_id=tab['pix_id'],
            pix_x=tab['pix_x'].quantity,
            pix_y=tab['pix_y'].quantity,
            pix_area=tab['pix_area'].quantity,
            pix_type=tab.meta['PIX_TYPE'],
            pix_rotation=Angle(tab.meta['PIX_ROT'] * u.deg),
            cam_rotation=Angle(tab.meta['CAM_ROT'] * u.deg),
        )

    def __str__(self):
        tab = self.to_table()
        return "CameraGeometry(cam_id='{cam_id}', pix_type='{pix_type}', " \
               "npix={npix})".format(cam_id=self.cam_id,
                                     pix_type=self.pix_type,
                                     npix=len(self.pix_id))

    @lazyproperty
    def neighbors(self):
        """" only calculate neighbors when needed or if not already 
        calculated"""

        # return pre-calculated ones (e.g. those that were passed in during
        # the object construction) if they exist
        if self._precalculated_neighbors is not None:
            return self._precalculated_neighbors

        # otherwise compute the neighbors from the pixel list
        dist = _get_min_pixel_seperation(self.pix_x, self.pix_y)

        neighbors = _find_neighbor_pixels(
            self.pix_x.value,
            self.pix_y.value,
            rad=1.4 * dist.value
        )

        return neighbors

    @lazyproperty
    def neighbor_matrix(self):
        return _neighbor_list_to_matrix(self.neighbors)

    def rotate(self, angle):
        """rotate the camera coordinates about the center of the camera by
        specified angle. Modifies the CameraGeometry in-place (so
        after this is called, the pix_x and pix_y arrays are
        rotated.

        Notes
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

    @classmethod
    def make_rectangular(cls, npix_x=40, npix_y=40, range_x=(-0.5, 0.5),
                         range_y=(-0.5, 0.5)):
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

        return cls(cam_id=-1,
                   pix_id=ids,
                   pix_x=xx * u.m,
                   pix_y=yy * u.m,
                   pix_area=(2 * rr) ** 2,
                   neighbors=None,
                   pix_type='rectangular')


# ======================================================================
# utility functions:
# ======================================================================

def _get_min_pixel_seperation(pix_x, pix_y):
    """
    Obtain the minimum seperation between two pixels on the camera

    Parameters
    ----------
    pix_x : array_like
        x position of each pixel
    pix_y : array_like
        y position of each pixels

    Returns
    -------
    pixsep : astropy.units.Unit

    """
    #    dx = pix_x[1] - pix_x[0]    <=== Not adjacent for DC-SSTs!!
    #    dy = pix_y[1] - pix_y[0]

    dx = pix_x - pix_x[0]
    dy = pix_y - pix_y[0]
    pixsep = np.min(np.sqrt(dx ** 2 + dy ** 2)[1:])
    return pixsep


def _find_neighbor_pixels(pix_x, pix_y, rad):
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
    global _CAMERA_GEOMETRY_TABLE

    try:
        return _CAMERA_GEOMETRY_TABLE[(npix, None)]
    except KeyError:
        return _CAMERA_GEOMETRY_TABLE.get((npix, round(optical_foclen.value, 1)),
                                          ('unknown', 'unknown', 'hexagonal',
                                  0 * u.degree, 0 * u.degree))


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
        neigh = _find_neighbor_pixels(xx.value, yy.value,
                                      (dd.mean() + 0.01 * u.m).value)

    return CameraGeometry(
        cam_id="{}:{}".format(instrument_name, cam_id),
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


def _neighbor_list_to_matrix(neighbors):
    """ 
    convert a neighbor adjacency list (list of list of neighbors) to a 2D 
    numpy array, which is much faster (and can simply be multiplied)
    """

    npix = len(neighbors)
    neigh2d = np.zeros(shape=(npix, npix), dtype=np.bool)

    for ipix, neighbors in enumerate(neighbors):
        for jn, neighbor in enumerate(neighbors):
            neigh2d[ipix, neighbor] = True

    return neigh2d
