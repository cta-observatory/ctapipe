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
from scipy.sparse import lil_matrix, csr_matrix
import warnings

from ctapipe.utils import get_table_dataset, find_all_matching_datasets
from ctapipe.utils.linalg import rotation_matrix_2d


__all__ = ['CameraGeometry']

logger = logging.getLogger(__name__)


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

        if pix_x.ndim != 1 or pix_y.ndim != 1:
            raise ValueError(f'Pixel coordinates must be 1 dimensional, got {pix_x.ndim}')

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
        self._neighbors = neighbors

        if neighbors is not None:
            if isinstance(neighbors, list):
                lil = lil_matrix((self.n_pixels, self.n_pixels), dtype=bool)
                for pix_id, neighbors in enumerate(neighbors):
                    lil[pix_id, neighbors] = True
                self._neighbors = lil.tocsr()
            else:
                self._neighbors = csr_matrix(neighbors)

        if self.pix_area is None:
            self.pix_area = CameraGeometry._calc_pixel_area(
                pix_x, pix_y, pix_type
            )

        if apply_derotation:
            # todo: this should probably not be done, but need to fix
            # GeometryConverter and reco algorithms if we change it.
            self.rotate(cam_rotation)

        # cache border pixel mask per instance
        self.border_cache = {}

    def __eq__(self, other):
        if self.cam_id != other.cam_id:
            return False

        if self.n_pixels != other.n_pixels:
            return False

        if self.pix_type != other.pix_type:
            return False

        if self.pix_rotation != other.pix_rotation:
            return False

        return all([
            (self.pix_x == other.pix_x).all(),
            (self.pix_y == other.pix_y).all(),
        ])

    def __hash__(self):
        return hash((
            self.cam_id,
            self.pix_x[0].to_value(u.m),
            self.pix_y[0].to_value(u.m),
            self.pix_type,
            self.pix_rotation.deg,
        ))

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

    @staticmethod
    def _calc_pixel_area(pix_x, pix_y, pix_type):
        """ recalculate pixel area based on the pixel type and layout

        Note this will not work on cameras with varying pixel sizes.
        """

        dist = np.min(np.sqrt((pix_x - pix_x[0])**2 + (pix_y - pix_y[0])**2))

        if pix_type.startswith('hex'):
            rad = dist / np.sqrt(3)  # radius to vertex of hexagon
            area = rad ** 2 * (3 * np.sqrt(3) / 2.0)  # area of hexagon
        elif pix_type.startswith('rect'):
            area = dist ** 2
        else:
            raise KeyError("unsupported pixel type")

        return np.ones(pix_x.shape) * area

    @lazyproperty
    def _pixel_circumferences(self):
        """ pixel circumference radius/radii based on pixel area and layout

        """

        if self.pix_type.startswith('hex'):
            circum_rad = np.sqrt(2.0 * self.pix_area / 3.0 / np.sqrt(3))
        elif self.pix_type.startswith('rect'):
            circum_rad = np.sqrt(self.pix_area / 2.0)
        else:
            raise KeyError("unsupported pixel type")

        return circum_rad

    @lazyproperty
    def _kdtree(self):
        """
        Pre-calculated kdtree of all pixel centers inside camera

        Returns
        -------
        kdtree

        """

        pixel_centers = np.column_stack([self.pix_x.to_value(u.m),
                                         self.pix_y.to_value(u.m)])
        return KDTree(pixel_centers)

    @lazyproperty
    def _all_pixel_areas_equal(self):
        """
        Pre-calculated kdtree of all pixel centers inside camera

        Returns
        -------
        True if all pixels are of equal size, False otherwise

        """
        return ~np.any(~np.isclose(self.pix_area.value, self.pix_area[0].value), axis=0)

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
            verstr = f"-{version:03d}"

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
        '''A list of the neighbors pixel_ids for each pixel'''
        return [np.where(r)[0].tolist() for r in self.neighbor_matrix]

    @lazyproperty
    def neighbor_matrix(self):
        return self.neighbor_matrix_sparse.A

    @lazyproperty
    def neighbor_matrix_sparse(self):
        if self._neighbors is not None:
            return self._neighbors
        else:
            return self.calc_pixel_neighbors(diagonal=False)

    def calc_pixel_neighbors(self, diagonal=False):
        '''
        Calculate the neighbors of pixels using
        a kdtree for nearest neighbor lookup.

        Parameters
        ----------
        diagonal: bool
            If rectangular geometry, also add diagonal neighbors
        '''
        neighbors = lil_matrix((self.n_pixels, self.n_pixels), dtype=bool)

        if self.pix_type.startswith('hex'):
            max_neighbors = 6
            # on a hexgrid, the closest pixel in the second circle is
            # the diameter of the hexagon plus the inradius away
            # in units of the diameter, this is 1 + np.sqrt(3) / 4 = 1.433
            radius = 1.4
            norm = 2  # use L2 norm for hex
        else:

            # if diagonal should count as neighbor, we
            # need to find at most 8 neighbors with a max L2 distance
            # < than 2 * the pixel size, else 4 neigbors with max L1 distance
            # < 2 pixel size. We take a conservative 1.5 here,
            # because that worked on the PROD4 CHEC camera that has
            # irregular pixel positions.
            if diagonal:
                max_neighbors = 8
                norm = 2
                radius = 1.95
            else:
                max_neighbors = 4
                radius = 1.5
                norm = 1

        for i, pixel in enumerate(self._kdtree.data):
            # as the pixel itself is in the tree, look for max_neighbors + 1
            distances, neighbor_candidates = self._kdtree.query(
                pixel, k=max_neighbors + 1, p=norm
            )

            # remove self-reference
            distances = distances[1:]
            neighbor_candidates = neighbor_candidates[1:]

            # remove too far away pixels
            inside_max_distance = distances < radius * np.min(distances)
            neighbors[i, neighbor_candidates[inside_max_distance]] = True

        # filter annoying deprecation warning from within scipy
        # scipy still uses np.matrix in scipy.sparse, but we do not
        # explicitly use any feature of np.matrix, so we can ignore this here
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=PendingDeprecationWarning)
            if (neighbors.T != neighbors).sum() > 0:
                warnings.warn(
                    'Neighbor matrix is not symmetric. Is camera geometry irregular?'
                )

        return neighbors.tocsr()

    @lazyproperty
    def neighbor_matrix_where(self):
        """
        Obtain a 2D array, where each row is [pixel index, one neighbour
        of that pixel].

        Returns
        -------
        ndarray
        """
        coo = self.neighbor_matrix_sparse.tocoo()
        return np.column_stack([coo.row, coo.col])

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
        printer(f'CameraGeometry: "{self}"')
        printer('   - num-pixels: {}'.format(len(self.pix_id)))
        printer(f'   - pixel-type: {self.pix_type}')
        printer('   - sensitive-area: {}'.format(self.pix_area.sum()))
        printer(f'   - pix-rotation: {self.pix_rotation}')
        printer(f'   - cam-rotation: {self.cam_rotation}')

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
                   pix_x=xx,
                   pix_y=yy,
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

        # filter annoying deprecation warning from within scipy
        # scipy still uses np.matrix in scipy.sparse, but we do not
        # explicitly use any feature of np.matrix, so we can ignore this here
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=PendingDeprecationWarning)

            if width == 1:
                n_neighbors = self.neighbor_matrix_sparse.sum(axis=1).A1
                max_neighbors = n_neighbors.max()
                mask = n_neighbors < max_neighbors
            else:
                n = self.neighbor_matrix
                mask = (n & self.get_border_pixel_mask(width - 1)).any(axis=1)

        self.border_cache[width] = mask
        return mask

    def position_to_pix_index(self, x, y):
        '''
        Return the index of a camera pixel which contains a given position (x,y)
        in the camera frame. The (x,y) coordinates can be arrays (of equal length),
        for which the methods returns an array of pixel ids. A warning is raised if the
        position falls outside the camera.

        Parameters
        ----------
        x: astropy.units.Quantity (distance) of horizontal position(s) in the camera frame
        y: astropy.units.Quantity (distance) of vertical position(s) in the camera frame

        Returns
        -------
        pix_indices: Pixel index or array of pixel indices. Returns -1 if position falls
                    outside camera
        '''

        if not self._all_pixel_areas_equal:
            logger.warning(" Method not implemented for cameras with varying pixel sizes")

        points_searched = np.dstack([x.to_value(u.m), y.to_value(u.m)])
        circum_rad = self._pixel_circumferences[0].to_value(u.m)
        kdtree = self._kdtree
        dist, pix_indices = kdtree.query(points_searched, distance_upper_bound=circum_rad)
        del dist
        pix_indices = pix_indices.flatten()

        # 1. Mark all points outside pixel circumeference as lying outside camera
        pix_indices[pix_indices == self.n_pixels] = -1

        # 2. Accurate check for the remaing cases (within circumference, but still outside
        # camera). It is first checked if any border pixel numbers are returned.
        # If not, everything is fine. If yes, the distance of the given position to the
        # the given position to the closest pixel center is translated to the distance to
        # the center of a non-border pixel', pos -> pos', and it is checked whether pos'
        # still lies within pixel'. If not, pos lies outside the camera. This approach
        # does not need to know the particular pixel shape, but as the kdtree itself,
        # presumes all camera pixels being of equal size.
        border_mask = self.get_border_pixel_mask()
        # get all pixels at camera border:
        borderpix_indices = np.where(border_mask)[0]
        borderpix_indices_in_list = np.intersect1d(borderpix_indices, pix_indices)
        if borderpix_indices_in_list.any():
            # Get some pixel not at the border:
            insidepix_index = np.where(~border_mask)[0][0]
            # Check in detail whether location is in border pixel or outside camera:
            for borderpix_index in borderpix_indices_in_list:
                index = np.where(pix_indices == borderpix_index)[0][0]
                # compare with inside pixel:
                xprime = (points_searched[0][index, 0]
                          - self.pix_x[borderpix_index].to_value(u.m)
                          + self.pix_x[insidepix_index].to_value(u.m))
                yprime = (points_searched[0][index, 1]
                          - self.pix_y[borderpix_index].to_value(u.m)
                          + self.pix_y[insidepix_index].to_value(u.m))
                dist_check, index_check = kdtree.query([xprime, yprime],
                                                       distance_upper_bound=circum_rad)
                del dist_check
                if index_check != insidepix_index:
                    pix_indices[index] = -1

        # print warning:
        for index in np.where(pix_indices == -1)[0]:
            logger.warning(" Coordinate ({} m, {} m) lies outside camera"
                           .format(points_searched[0][index, 0],
                                   points_searched[0][index, 1]))

        return pix_indices if len(pix_indices) > 1 else pix_indices[0]

    @staticmethod
    def simtel_shape_to_type(pixel_shape):
        if pixel_shape == 1:
            return 'hexagonal', Angle(0, u.deg)

        if pixel_shape == 2:
            return 'rectangular', Angle(0, u.deg)

        if pixel_shape == 3:
            return 'hexagonal', Angle(30, u.deg)

        raise ValueError(f'Unknown pixel_shape {pixel_shape}')


class UnknownPixelShapeWarning(UserWarning):
    pass
