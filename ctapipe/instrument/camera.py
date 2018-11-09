# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utilities for reading or working with Camera geometry files
"""
import logging

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from astropy.table import Table
from astropy.utils import lazyproperty
from scipy.spatial import cKDTree as KDTree
from scipy.sparse import csr_matrix

from ctapipe.utils import get_table_dataset, find_all_matching_datasets
from ctapipe.utils.linalg import rotation_matrix_2d


__all__ = ['CameraGeometry']

logger = logging.getLogger(__name__)

# dictionary to convert number of pixels to camera + the focal length of the
# telescope into a camera type for use in `CameraGeometry.guess()`
#     Key = (num_pix, focal_length_in_meters)
#     Value = (type, subtype, pixtype, pixrotation, camrotation)
_CAMERA_GEOMETRY_TABLE = {
    (2048, 2.3): ('SST', 'CHEC', 'rectangular', 0 * u.degree, 0 * u.degree),
    (2048, 2.2): ('SST', 'CHEC', 'rectangular', 0 * u.degree, 0 * u.degree),
    (2048, 36.0): ('LST', 'HESS-II', 'hexagonal', 0 * u.degree,
                   0 * u.degree),
    (960, None): ('MST', 'HESS-I', 'hexagonal', 0 * u.degree,
                  0 * u.degree),
    (1855, 16.0): ('MST', 'NectarCam', 'hexagonal',
                   0 * u.degree, -100.893 * u.degree),
    (1855, 28.0): ('LST', 'LSTCam', 'hexagonal',
                   0. * u.degree, -100.893 * u.degree),
    (1296, None): ('SST', 'DigiCam', 'hexagonal', 30 * u.degree, 0 * u.degree),
    (1764, None): ('MST', 'FlashCam', 'hexagonal', 30 * u.degree, 0 * u.degree),
    (2368, None): ('SST', 'ASTRICam', 'rectangular', 0 * u.degree,
                   0 * u.degree),
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
        surface area of each pixel, if None will be calculated
    neighbors: list(arrays)
        adjacency list for each pixel
    pix_type: string
        either 'rectangular' or 'hexagonal'
    pix_rotation: value convertable to an `astropy.coordinates.Angle`
        rotation angle with unit (e.g. 12 * u.deg), or "12d"
    cam_rotation: overall camera rotation with units
    """

    _geometry_cache = {}  # dictionary CameraGeometry instances for speed

    def __init__(self, cam_id, pix_id, pix_x, pix_y, pix_area, pix_type,
                 pix_rotation="0d", cam_rotation="0d",
                 neighbors=None, apply_derotation=True):

        assert len(pix_x) == len(pix_y), 'pix_x and pix_y must have same length'
        self.n_pixels = len(pix_x)
        self.cam_id = cam_id
        self.pix_id = pix_id
        self.pix_x = pix_x
        self.pix_y = pix_y
        self.pix_area = pix_area
        self.pix_type = pix_type
        self.pix_rotation = Angle(pix_rotation)
        self.cam_rotation = Angle(cam_rotation)
        self._precalculated_neighbors = neighbors

        if self.pix_area is None:
            self.pix_area = CameraGeometry._calc_pixel_area(pix_x, pix_y,
                                                            pix_type)

        if apply_derotation:
            # todo: this should probably not be done, but need to fix
            # GeometryConverter and reco algorithms if we change it.
            if len(pix_x.shape) == 1:
                self.rotate(cam_rotation)

        # cache border pixel mask per instance
        self.border_cache = {}

    def __eq__(self, other):
        return ((self.cam_id == other.cam_id)
                and (self.pix_x == other.pix_x).all()
                and (self.pix_y == other.pix_y).all()
                and (self.pix_type == other.pix_type)
                and (self.pix_rotation == other.pix_rotation)
                and (self.pix_type == other.pix_type)
                )

    def __len__(self):
        return self.n_pixels

    def __getitem__(self, slice_):
        return CameraGeometry(
            cam_id=" ".join([self.cam_id, " sliced"]),
            pix_id=self.pix_id[slice_],
            pix_x=self.pix_x[slice_],
            pix_y=self.pix_y[slice_],
            pix_area=self.pix_area[slice_],
            pix_type=self.pix_type,
            pix_rotation=self.pix_rotation,
            cam_rotation=self.cam_rotation,
            neighbors=None,
            apply_derotation=False,
        )

    @classmethod
    @u.quantity_input
    def guess(cls, pix_x: u.m, pix_y: u.m, optical_foclen: u.m,
              apply_derotation=True):
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

        tel_type, cam_id, pix_type, pix_rotation, cam_rotation = \
            _guess_camera_type(len(pix_x), optical_foclen)

        area = cls._calc_pixel_area(pix_x, pix_y, pix_type)

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
            apply_derotation=apply_derotation
        )
        instance.cam_rotation = Angle(cam_rotation)
        CameraGeometry._geometry_cache[identifier] = instance
        return instance

    @staticmethod
    def _calc_pixel_area(pix_x, pix_y, pix_type):
        """ recalculate pixel area based on the pixel type and layout

        Note this will not work on cameras with varying pixel sizes.
        """

        dist = _get_min_pixel_seperation(pix_x, pix_y)

        if pix_type.startswith('hex'):
            rad = dist / np.sqrt(3)  # radius to vertex of hexagon
            area = rad ** 2 * (3 * np.sqrt(3) / 2.0)  # area of hexagon
        elif pix_type.startswith('rect'):
            area = dist ** 2
        else:
            raise KeyError("unsupported pixel type")

        return np.ones(pix_x.shape) * area

    @classmethod
    def get_known_camera_names(cls):
        """
        Returns a list of camera_ids that are registered in
        `ctapipe_resources`. These are all the camera-ids that can be
        instantiated by the `from_name` method

        Returns
        -------
        list(str)
        """

        pattern = r'(.*)\.camgeom\.fits(\.gz)?'
        return find_all_matching_datasets(pattern, regexp_group=1)

    @classmethod
    def from_name(cls, camera_id='NectarCam', version=None):
        """
        Construct a CameraGeometry using the name of the camera and array.

        This expects that there is a resource in the `ctapipe_resources` module
        called "[array]-[camera].camgeom.fits.gz" or "[array]-[camera]-[
        version].camgeom.fits.gz"

        Parameters
        ----------
        camera_id: str
           name of camera (e.g. 'NectarCam', 'LSTCam', 'GCT', 'SST-1M')
        version:
           camera version id (currently unused)

        Returns
        -------
        new CameraGeometry
        """

        if version is None:
            verstr = ''
        else:
            verstr = "-{:03d}".format(version)

        tabname = "{camera_id}{verstr}.camgeom".format(camera_id=camera_id,
                                                       verstr=verstr)
        table = get_table_dataset(tabname, role='dl0.tel.svc.camera')
        return CameraGeometry.from_table(table)

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
        kwargs: extra keyword arguments
            extra arguments passed to `astropy.table.read()`, depending on
            file type (e.g. format, hdu, path)


        """

        tab = url_or_table
        if not isinstance(url_or_table, Table):
            tab = Table.read(url_or_table, **kwargs)

        return cls(
            cam_id=tab.meta.get('CAM_ID', 'Unknown'),
            pix_id=tab['pix_id'],
            pix_x=tab['pix_x'].quantity,
            pix_y=tab['pix_y'].quantity,
            pix_area=tab['pix_area'].quantity,
            pix_type=tab.meta['PIX_TYPE'],
            pix_rotation=Angle(tab.meta['PIX_ROT'] * u.deg),
            cam_rotation=Angle(tab.meta['CAM_ROT'] * u.deg),
        )

    def __repr__(self):
        return (
            "CameraGeometry(cam_id='{cam_id}', pix_type='{pix_type}', "
            "npix={npix}, cam_rot={camrot}, pix_rot={pixrot})"
        ).format(
            cam_id=self.cam_id,
            pix_type=self.pix_type,
            npix=len(self.pix_id),
            pixrot=self.pix_rotation,
            camrot=self.cam_rotation
        )

    def __str__(self):
        return self.cam_id

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

    @lazyproperty
    def neighbor_matrix_sparse(self):
        return csr_matrix(self.neighbor_matrix)

    @lazyproperty
    def neighbor_matrix_where(self):
        """
        Obtain a 2D array, where each row is [pixel index, one neighbour
        of that pixel].

        Returns
        -------
        ndarray
        """
        return np.ascontiguousarray(np.array(np.where(self.neighbor_matrix)).T)

    @lazyproperty
    def pixel_moment_matrix(self):
        """
        Pre-calculated matrix needed for higher-order moment calculation,
        up to 4th order.

        Note this is *not* recalculated if the CameraGeometry is modified.

        this matrix M can be multiplied by an image and normalized by the sum to
        get the moments:

        .. code-block:: python3

            M = geom.pixel_moment_matrix()
            moms = (M @ image)/image.sum()


        Returns
        -------
        array:
            x, y, x**2, x*y, y^2, x^3, x^2*y,x*y^2, y^3, x^4, x^3*y, x^2*y2,
            x*y^3, y^4

        """

        x = self.pix_x.value
        y = self.pix_y.value

        return np.row_stack([x, y,
                             x ** 2, x * y, y ** 2,
                             x ** 3, x ** 2 * y, x * y ** 2, y ** 3,
                             x ** 4, x ** 3 * y, x ** 2 * y ** 2, x * y ** 3,
                             y ** 4])

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
        self.pix_rotation -= Angle(angle)
        self.cam_rotation -= Angle(angle)

    def info(self, printer=print):
        """ print detailed info about this camera """
        printer('CameraGeometry: "{}"'.format(self))
        printer('   - num-pixels: {}'.format(len(self.pix_id)))
        printer('   - pixel-type: {}'.format(self.pix_type))
        printer('   - sensitive-area: {}'.format(self.pix_area.sum()))
        printer('   - pix-rotation: {}'.format(self.pix_rotation))
        printer('   - cam-rotation: {}'.format(self.cam_rotation))

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

    def get_border_pixel_mask(self, width=1):
        '''
        Get a mask for pixels at the border of the camera of arbitrary width

        Parameters
        ----------
        width: int
            The width of the border in pixels

        Returns
        -------
        mask: array
            A boolean mask, True if pixel is in the border of the specified width
        '''
        if width in self.border_cache:
            return self.border_cache[width]

        if width == 1:
            n_neighbors = self.neighbor_matrix_sparse.sum(axis=1).A1
            max_neighbors = n_neighbors.max()
            mask = n_neighbors < max_neighbors
        else:
            n = self.neighbor_matrix
            mask = (n & self.get_border_pixel_mask(width - 1)).any(axis=1)

        self.border_cache[width] = mask
        return mask

    def get_shower_coordinates(self, x, y, psi):
        '''
        Return longitudinal and transverse coordinates of the pixels
        for a given set of hillas parameters

        Parameters
        ----------
        hillas_parameters: ctapipe.io.containers.HilllasContainer

        Returns
        -------
        longitudinal: astropy.units.Quantity
            longitudinal coordinates (along the shower axis)
        transverse: astropy.units.Quantity
            transverse coordinates (perpendicular to the shower axis)
        '''
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)

        delta_x = self.pix_x - x
        delta_y = self.pix_y - y

        longi = delta_x * cos_psi + delta_y * sin_psi
        trans = delta_x * -sin_psi + delta_y * cos_psi

        return longi, trans

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
        return _CAMERA_GEOMETRY_TABLE.get(
            (npix, round(optical_foclen.value, 1)),
            ('unknown', 'unknown', 'hexagonal',
             0 * u.degree, 0 * u.degree)
        )


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
