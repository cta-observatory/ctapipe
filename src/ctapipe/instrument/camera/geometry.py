# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utilities for reading or working with Camera geometry files
"""
import logging
import warnings
from copy import deepcopy
from enum import Enum, unique

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle, BaseCoordinateFrame, SkyCoord
from astropy.table import Table
from astropy.utils import lazyproperty
from scipy.sparse import csr_matrix, lil_matrix
from scipy.spatial import cKDTree

from ctapipe.coordinates import CameraFrame, get_representation_component_names
from ctapipe.utils import get_table_dataset
from ctapipe.utils.linalg import rotation_matrix_2d

from ..warnings import warn_from_name
from .image_conversion import (
    get_orthogonal_grid_edges,
    get_orthogonal_grid_indices,
    unskew_hex_pixel_grid,
)

__all__ = ["CameraGeometry", "UnknownPixelShapeWarning", "PixelShape"]

logger = logging.getLogger(__name__)


def _distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


@unique
class PixelShape(Enum):
    """Supported Pixel Shapes Enum"""

    CIRCLE = "circle"
    SQUARE = "square"
    HEXAGON = "hexagon"

    @classmethod
    def from_string(cls, name):
        """
        Convert a string representation to the enum value

        This function supports abbreviations and for backwards compatibility
        "rect" as alias for "square".
        """
        name = name.lower()

        if name.startswith("hex"):
            return cls.HEXAGON

        if name.startswith("rect") or name == "square":
            return cls.SQUARE

        if name.startswith("circ"):
            return cls.CIRCLE

        raise TypeError(f"Unknown pixel shape {name}")


#: mapper from simtel pixel shape integers to our shape and rotation angle
SIMTEL_PIXEL_SHAPES = {
    0: (PixelShape.CIRCLE, Angle(0, u.deg)),
    1: (PixelShape.HEXAGON, Angle(0, u.deg)),
    2: (PixelShape.SQUARE, Angle(0, u.deg)),
    3: (PixelShape.HEXAGON, Angle(30, u.deg)),
}


class CameraGeometry:
    """`CameraGeometry` is a class that stores information about a
    Cherenkov Camera that is useful for imaging algorithms and
    displays. It contains lists of pixel positions, areas, pixel
    shapes, as well as a neighbor (adjacency) list and matrix for each pixel.
    In general the neighbor_matrix attribute should be used in any algorithm
    needing pixel neighbors, since it is much faster. See for example
    `ctapipe.image.tailcuts_clean`

    The class is intended to be generic, and work with any Cherenkov
    Camera geometry, including those that have square vs hexagonal
    pixels, gaps between pixels, etc.

    Parameters
    ----------
    name : str
         Camera name (e.g. "NectarCam", "LSTCam", ...)
    pix_id : np.ndarray[int]
        pixels id numbers
    pix_x : u.Quantity
        position of each pixel (x-coordinate)
    pix_y : u.Quantity
        position of each pixel (y-coordinate)
    pix_area : u.Quantity
        surface area of each pixel
    pix_type : PixelShape
        either 'rectangular' or 'hexagonal'
    pix_rotation : u.Quantity[angle]
        rotation of the pixels, global value for all pixels
    cam_rotation : u.Quantity[angle]
        rotation of the camera. All coordinates will be interpreted
        to be rotated by this angle with respect to the definition of the
        frame the pixel coordinates are defined in.
    neighbors : list[list[int]]
        adjacency list for each pixel
        If not given, will be build from the pixel width and distances
        between pixels.
    apply_derotation : bool
        If true, rotate the pixel coordinates, so that cam_rotation is 0.
    frame : coordinate frame
        Frame in which the pixel coordinates are defined (after applying cam_rotation)
    """

    CURRENT_TAB_VERSION = "2.0"
    SUPPORTED_TAB_VERSIONS = {"1.0", "1", "1.1", "2.0"}

    def __init__(
        self,
        name,
        pix_id,
        pix_x,
        pix_y,
        pix_area,
        pix_type,
        pix_rotation=0 * u.deg,
        cam_rotation=0 * u.deg,
        neighbors=None,
        apply_derotation=True,
        frame=None,
        _validate=True,
    ):
        self.name = name
        self.n_pixels = len(pix_id)
        self.unit = pix_x.unit
        self.pix_id = np.array(pix_id)

        if _validate:
            self.pix_id = np.array(self.pix_id)

            if self.pix_id.ndim != 1:
                raise ValueError(
                    f"Pixel coordinates must be 1 dimensional, got {pix_id.ndim}"
                )

            shape = (self.n_pixels,)

            if pix_x.shape != shape:
                raise ValueError(
                    f"pix_x has wrong shape: {pix_x.shape}, expected {shape}"
                )
            if pix_y.shape != shape:
                raise ValueError(
                    f"pix_y has wrong shape: {pix_y.shape}, expected {shape}"
                )
            if pix_area.shape != shape:
                raise ValueError(
                    f"pix_area has wrong shape: {pix_area.shape}, expected {shape}"
                )

            if isinstance(pix_type, str):
                pix_type = PixelShape.from_string(pix_type)
            elif not isinstance(pix_type, PixelShape):
                raise TypeError(
                    f"pix_type must be a PixelShape or the name of a PixelShape, got {pix_type}"
                )

            if not isinstance(pix_rotation, Angle):
                pix_rotation = Angle(pix_rotation)

            if not isinstance(cam_rotation, Angle):
                cam_rotation = Angle(cam_rotation)

        self.pix_x = pix_x
        self.pix_y = pix_y.to(self.unit)
        self.pix_area = pix_area.to(self.unit**2)
        self.pix_type = pix_type
        self.pix_rotation = pix_rotation
        self.cam_rotation = cam_rotation
        self._neighbors = neighbors
        self.frame = frame

        if neighbors is not None:
            if isinstance(neighbors, list):
                lil = lil_matrix((self.n_pixels, self.n_pixels), dtype=bool)
                for pix_id, neighbors in enumerate(neighbors):
                    lil[pix_id, neighbors] = True
                self._neighbors = lil.tocsr()
            else:
                self._neighbors = csr_matrix(neighbors)

        if apply_derotation:
            self.rotate(self.cam_rotation)

            # cache border pixel mask per instance
        self._border_cache = {}

    def __eq__(self, other):
        if not isinstance(other, CameraGeometry):
            return NotImplemented

        if self.name != other.name:
            return False

        if self.n_pixels != other.n_pixels:
            return False

        if self.pix_type != other.pix_type:
            return False

        if not u.isclose(self.pix_rotation, other.pix_rotation):
            return False

        return all(
            [u.allclose(self.pix_x, other.pix_x), u.allclose(self.pix_y, other.pix_y)]
        )

    def guess_radius(self):
        """
        Guess the camera radius as mean distance of the border pixels from
        the center pixel
        """
        border = self.get_border_pixel_mask()
        cx = self.pix_x.mean()
        cy = self.pix_y.mean()

        return np.sqrt(
            (self.pix_x[border] - cx) ** 2 + (self.pix_y[border] - cy) ** 2
        ).mean()

    def transform_to(self, frame: BaseCoordinateFrame):
        """Transform the pixel coordinates stored in this geometry and the pixel
        and camera rotations to another camera coordinate frame.

        Note that ``geom.frame`` must contain all the necessary attributes needed
        to transform into the requested frame, i.e. if going from
        `~ctapipe.coordinates.CameraFrame` to `~ctapipe.coordinates.TelescopeFrame`,
        it should contain the correct data in the ``focal_length`` attribute.

        Parameters
        ----------
        frame: ctapipe.coordinates.CameraFrame
            The coordinate frame to transform to.

        Returns
        -------
        CameraGeometry:
            new instance in the requested Frame
        """
        if self.frame is None:
            self.frame = CameraFrame()

        # we have to derotate, transformations only provide sensible
        # results after derotation from the camera coordinate system with
        # custom angle into a not-rotate frame
        if self.cam_rotation.value != 0:
            cam = deepcopy(self)
            cam.rotate(self.cam_rotation)
        else:
            cam = self

        coord = SkyCoord(cam.pix_x, cam.pix_y, frame=cam.frame)
        trans = coord.transform_to(frame)

        # also transform the origin and unit vectors,
        # needed to account for translation, rotation / mirroring, scale
        width = cam.pixel_width[0].to_value(cam.pix_x.unit)
        points = SkyCoord(
            [0, width, 0], [0, 0, width], unit=cam.pix_x.unit, frame=cam.frame
        )
        points_trans = points.transform_to(frame)

        x_name, y_name = get_representation_component_names(cam.frame)
        points_x = getattr(points, x_name)
        points_y = getattr(points, y_name)

        trans_x_name, trans_y_name = get_representation_component_names(frame)
        points_trans_x = getattr(points_trans, trans_x_name)
        points_trans_y = getattr(points_trans, trans_y_name)

        matrix = np.vstack([points_trans_x[1:].value, points_trans_y[1:].value])
        is_mirrored = np.linalg.det(matrix) < 0

        rot = np.arctan2(
            points_trans_y[1] - points_trans_y[0], points_trans_y[2] - points_trans_y[0]
        )

        if is_mirrored:
            cam_rotation = -cam.cam_rotation
            pix_rotation = rot - cam.pix_rotation
        else:
            cam_rotation = cam.cam_rotation
            pix_rotation = cam.pix_rotation - rot

        distance_before = _distance(
            points_x[1],
            points_y[1],
            points_x[2],
            points_y[2],
        )
        distance_after = _distance(
            points_trans_x[1],
            points_trans_y[1],
            points_trans_x[2],
            points_trans_y[2],
        )
        scale = distance_after / distance_before

        trans_x = getattr(trans, trans_x_name)
        trans_y = getattr(trans, trans_y_name)
        pix_area = (cam.pix_area * scale**2).to(trans_x.unit**2)

        return CameraGeometry(
            name=cam.name,
            pix_id=cam.pix_id,
            pix_x=trans_x,
            pix_y=trans_y,
            pix_area=pix_area,
            pix_type=cam.pix_type,
            pix_rotation=pix_rotation,
            cam_rotation=cam_rotation,
            neighbors=cam._neighbors,
            apply_derotation=False,
            frame=frame,
        )

    def __hash__(self):
        return hash(
            (
                self.name,
                round(self.pix_x[0].value, 3),
                round(self.pix_y[0].value, 3),
                self.pix_type,
                round(self.pix_rotation.deg, 3),
            )
        )

    def __len__(self):
        return self.n_pixels

    def __getitem__(self, slice_):
        return CameraGeometry(
            name=" ".join([self.name, " sliced"]),
            pix_id=self.pix_id[slice_],
            pix_x=self.pix_x[slice_],
            pix_y=self.pix_y[slice_],
            pix_area=self.pix_area[slice_],
            pix_type=self.pix_type,
            pix_rotation=self.pix_rotation,
            cam_rotation=self.cam_rotation,
            neighbors=None,
            apply_derotation=False,
            _validate=False,
        )

    @lazyproperty
    def pixel_width(self):
        """
        in-circle diameter for hexagons, edge width for square pixels,
        diameter for circles.

        This is calculated from the pixel area.
        """

        if self.pix_type == PixelShape.HEXAGON:
            width = 2 * np.sqrt(self.pix_area / (2 * np.sqrt(3)))
        elif self.pix_type == PixelShape.SQUARE:
            width = np.sqrt(self.pix_area)
        elif self.pix_type == PixelShape.CIRCLE:
            width = 2 * np.sqrt(self.pix_area / np.pi)
        else:
            raise NotImplementedError(
                f"Cannot calculate pixel width for type {self.pix_type!r}"
            )

        return width

    @staticmethod
    def guess_pixel_width(pix_x, pix_y):
        """
        Calculate pixel diameter by looking at the minimum distance between pixels

        Note this will not work on cameras with varying pixel sizes or gaps

        Returns
        -------
            in-circle diameter for hexagons, edge width for square pixels
        """
        return np.min(
            np.sqrt((pix_x[1:] - pix_x[0]) ** 2 + (pix_y[1:] - pix_y[0]) ** 2)
        )

    @lazyproperty
    def _pixel_circumradius(self):
        """pixel circumference radius/radii based on pixel area and layout"""

        if self.pix_type == PixelShape.HEXAGON:
            circum_rad = self.pixel_width / np.sqrt(3)
        elif self.pix_type == PixelShape.SQUARE:
            circum_rad = np.sqrt(self.pix_area / 2.0)
        elif self.pix_type == PixelShape.CIRCLE:
            circum_rad = self.pixel_width / 2
        else:
            raise NotImplementedError(
                "Cannot calculate pixel circumradius for type {self.pix_type!r}"
            )

        return circum_rad

    @lazyproperty
    def _kdtree(self):
        """
        Pre-calculated kdtree of all pixel centers inside camera

        Returns
        -------
        kdtree

        """

        pixel_centers = np.column_stack([self.pix_x.value, self.pix_y.value])
        return cKDTree(pixel_centers)

    @lazyproperty
    def _all_pixel_areas_equal(self):
        """
        Pre-calculated kdtree of all pixel centers inside camera

        Returns
        -------
        True if all pixels are of equal size, False otherwise

        """
        return ~np.any(~np.isclose(self.pix_area.value, self.pix_area[0].value), axis=0)

    def image_index_to_cartesian_index(self, pixel_index):
        """
        Convert pixel index in the 1d image representation to row and col
        """
        rows, cols = self._pixel_positions_2d
        return rows[pixel_index], cols[pixel_index]

    def cartesian_index_to_image_index(self, row, col):
        """
        Convert cartesian index (row, col) to pixel index in 1d representation.
        """
        return self._pixel_indices_cartesian[row, col]

    @lazyproperty
    def _pixel_indices_cartesian(self):
        img = np.arange(self.n_pixels)
        img2d = self.image_to_cartesian_representation(img)
        invalid = np.iinfo(np.int32).min
        img2d = np.nan_to_num(img2d, nan=invalid).astype(np.int32)
        return img2d

    @lazyproperty
    def _pixel_positions_2d(self):
        """
        Pixel positions on the orthogonal grid of the 2d image.
        In order for hexagonal pixels to behave as if they were
        square, the grid has to be distorted.
        Namely, slanting and stretching of the 1d pixel positions
        to align them nicely.
        Beware, that this means the pixel geometries on this
        grid to not match the one in the geometry.

        Returns
        -------
        (rows, columns) of each pixel if transformed onto an orthogonal grid
        """
        if self.pix_type in {PixelShape.HEXAGON, PixelShape.CIRCLE}:
            # cam rotation should be 0 unless the derotation is turned off in the init
            rot_x, rot_y = unskew_hex_pixel_grid(
                self.pix_x,
                self.pix_y,
                cam_angle=30 * u.deg - self.pix_rotation - self.cam_rotation,
            )
            x_edges, y_edges, _ = get_orthogonal_grid_edges(
                rot_x.to_value(u.m), rot_y.to_value(u.m)
            )
            square_mask = np.histogramdd(
                [rot_x.to_value(u.m), rot_y.to_value(u.m)], bins=(x_edges, y_edges)
            )[0].astype(bool)
            hex_to_rect_map = np.histogramdd(
                [rot_x.to_value(u.m), rot_y.to_value(u.m)],
                bins=(x_edges, y_edges),
                weights=np.arange(len(self.pix_y)),
            )[0].astype(int)
            hex_to_rect_map[~square_mask] = -1
            rows_2d = np.zeros(hex_to_rect_map.shape)
            rows_2d.T[:] = np.arange(hex_to_rect_map.shape[0])
            rows_1d = np.zeros(self.pix_x.shape, dtype=np.int32)
            rows_1d[hex_to_rect_map[..., square_mask]] = np.squeeze(
                np.rollaxis(np.atleast_3d(rows_2d), 2, 0)
            )[..., square_mask]
            cols_2d = np.zeros(hex_to_rect_map.shape)
            cols_2d[:] = np.arange(hex_to_rect_map.shape[1])
            cols_1d = np.zeros(self.pix_x.shape, dtype=np.int32)
            cols_1d[hex_to_rect_map[..., square_mask]] = np.squeeze(
                np.rollaxis(np.atleast_3d(cols_2d), 2, 0)
            )[..., square_mask]
            pixel_row = rows_1d
            pixel_column = cols_1d

            # flip image so that imshow looks like original camera display
            pixel_row = pixel_row.max() - pixel_row
            pixel_column = pixel_column.max() - pixel_column

        elif self.pix_type is PixelShape.SQUARE:
            pixel_row = get_orthogonal_grid_indices(self.pix_y, np.sqrt(self.pix_area))
            pixel_column = get_orthogonal_grid_indices(
                self.pix_x, np.sqrt(self.pix_area)
            )

            # flip image so that imshow looks like original camera display
            pixel_row = pixel_row.max() - pixel_row
        else:
            raise ValueError(f"Unsupported pixel shape {self.pix_type}")

        return pixel_row, pixel_column

    def image_to_cartesian_representation(self, image):
        """
        Create a 2D-image from a given flat image or multiple flat images.
        In the case of hexagonal pixels, the resulting
        image is skewed to match a rectangular grid.

        Parameters
        ----------
        image: np.ndarray
            One or multiple images of shape
            (n_images, n_pixels) or (n_pixels) for a single image.

        Returns
        -------
        image_2s: np.ndarray
            The transformed image of shape (n_images, n_rows, n_cols).
            For a single image the leading dimension is omitted.
        """
        rows, cols = self._pixel_positions_2d
        image = np.atleast_2d(image)  # this allows for multiple images at once

        image_2d = np.full((image.shape[0], rows.max() + 1, cols.max() + 1), np.nan)
        image_2d[:, rows, cols] = image

        return np.squeeze(image_2d)  # removes the extra dimension for single images

    def image_from_cartesian_representation(self, image_2d):
        """
        Create a 1D-array from a given 2D image.

        Parameters
        ----------
        image_2d: np.ndarray
            2D image created by the `image_to_cartesian_representation` function
            of the same geometry.
            shape is expected to be:
            (n_images, n_rows, n_cols) or (n_rows, n_cols) for a single image.

        Returns
        -------
        1d array
            The image in the 1D format, which has shape (n_images, n_pixels).
            For single images the leading dimension is omitted.
        """
        rows, cols = self._pixel_positions_2d
        # np.atleast3d would introduce the extra dimension at the end, which leads
        # to a different shape compared to a multi image array
        if image_2d.ndim == 2:
            image_2d = image_2d[np.newaxis, :]

        image_flat = np.zeros((image_2d.shape[0], rows.shape[0]), dtype=image_2d.dtype)
        image_flat[:] = image_2d[:, rows, cols]
        image_1d = image_flat
        return np.squeeze(image_1d)

    @classmethod
    def from_name(cls, name="NectarCam", version=None):
        """Construct a CameraGeometry using the name of the camera and array.

        This expects that there is a resource accessible via
        `~ctapipe.utils.get_table_dataset` called ``"[array]-[camera].camgeom.fits.gz"``
        or ``"[array]-[camera]-[version].camgeom.fits.gz"``

        Notes
        -----

        Warning: This method loads a pre-generated ``CameraGeometry`` and is
        thus not guaranteed to be the same pixel ordering or even positions that
        correspond with event data! Therefore if you are analysing data, you
        should not rely on this method, but rather open the data with an
        ``EventSource`` and use the ``CameraGeometry`` that is provided by
        ``source.subarray.tel[i].camera.geometry`` or by
        ``source.subarray.camera_types[type_name].geometry``. This will
        guarantee that the pixels in the event data correspond with the
        ``CameraGeometry``

        Parameters
        ----------
        name : str
            Camera name (e.g. NectarCam, LSTCam, ...)
        version :
            camera version id

        Returns
        -------
        new CameraGeometry

        """
        warn_from_name()

        if version is None:
            verstr = ""
        else:
            verstr = f"-{version:03d}"

        tabname = "{name}{verstr}.camgeom".format(name=name, verstr=verstr)
        table = get_table_dataset(tabname, role="dl0.tel.svc.camera")
        return CameraGeometry.from_table(table)

    def to_table(self):
        """convert this to an `astropy.table.Table`"""
        # currently the neighbor list is not supported, since
        # var-length arrays are not supported by astropy.table.Table
        t = Table(
            [self.pix_id, self.pix_x, self.pix_y, self.pix_area],
            names=["pix_id", "pix_x", "pix_y", "pix_area"],
            meta=dict(
                PIX_TYPE=self.pix_type.value,
                TAB_TYPE="ctapipe.instrument.CameraGeometry",
                TAB_VER=self.CURRENT_TAB_VERSION,
                CAM_ID=self.name,
                PIX_ROT=self.pix_rotation.deg,
                CAM_ROT=self.cam_rotation.deg,
            ),
        )

        # clear `info` member from quantities set by table creation
        # which impacts indexing performance because it is deepcopied
        # in Quantity.__getitem__, see https://github.com/astropy/astropy/issues/11066
        for q in (self.pix_id, self.pix_x, self.pix_y, self.pix_area):
            if hasattr(q, "__dict__"):
                if "info" in q.__dict__:
                    del q.__dict__["info"]

        return t

    @classmethod
    def from_table(cls, url_or_table, **kwargs):
        """
        Load a CameraGeometry from an `astropy.table.Table` instance or a
        file that is readable by `astropy.table.Table.read`

        Parameters
        ----------
        url_or_table: string or astropy.table.Table
            either input filename/url or a Table instance
        kwargs: extra keyword arguments
            extra arguments passed to `astropy.table.Table.read`, depending on
            file type (e.g. format, hdu, path)
        """

        tab = url_or_table
        if not isinstance(url_or_table, Table):
            tab = Table.read(url_or_table, **kwargs)

        version = tab.meta.get("TAB_VER")
        if version not in cls.SUPPORTED_TAB_VERSIONS:
            raise OSError(f"Unsupported camera geometry table version: {version}")

        return cls(
            name=tab.meta.get("CAM_ID", "Unknown"),
            pix_id=tab["pix_id"],
            pix_x=tab["pix_x"].quantity,
            pix_y=tab["pix_y"].quantity,
            pix_area=tab["pix_area"].quantity,
            pix_type=tab.meta["PIX_TYPE"],
            pix_rotation=Angle(tab.meta["PIX_ROT"], u.deg),
            cam_rotation=Angle(tab.meta["CAM_ROT"], u.deg),
        )

    def __repr__(self):
        return (
            "CameraGeometry(name='{name}', pix_type={pix_type}, "
            "npix={npix}, cam_rot={camrot:.3f}, pix_rot={pixrot:.3f}, frame={frame})"
        ).format(
            name=self.name,
            pix_type=self.pix_type,
            npix=len(self.pix_id),
            pixrot=self.pix_rotation,
            camrot=self.cam_rotation,
            frame=self.frame,
        )

    def __str__(self):
        return self.name

    @lazyproperty
    def neighbors(self):
        """A list of the neighbors pixel_ids for each pixel"""
        return [np.where(r)[0].tolist() for r in self.neighbor_matrix]

    @lazyproperty
    def neighbor_matrix(self):
        return self.neighbor_matrix_sparse.toarray()

    @lazyproperty
    def max_neighbors(self):
        return self.neighbor_matrix_sparse.sum(axis=1).max()

    @lazyproperty
    def neighbor_matrix_sparse(self):
        if self._neighbors is not None:
            return self._neighbors
        else:
            return self.calc_pixel_neighbors(diagonal=False)

    def calc_pixel_neighbors(self, diagonal=False):
        """
        Calculate the neighbors of pixels using
        a kdtree for nearest neighbor lookup.

        Parameters
        ----------
        diagonal: bool
            If rectangular geometry, also add diagonal neighbors
        """
        if self.n_pixels <= 1:
            return csr_matrix(np.ones((self.n_pixels, self.n_pixels), dtype=bool))

        # assume circle pixels are also on a hex grid
        if self.pix_type in (PixelShape.HEXAGON, PixelShape.CIRCLE):
            max_neighbors = 6
            # on a hexgrid, the closest pixel in the second circle is
            # the diameter of the hexagon plus the inradius away
            # in units of the diameter, this is 1 + np.sqrt(3) / 4 = 1.433
            radius = 1.4
            norm = 2  # use L2 norm for hex
        else:
            # if diagonal should count as neighbor, we
            # need to find at most 8 neighbors with a max L2 distance
            # < than 2 * the pixel size, else 4 neighbors with max L1 distance
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

        distances, neighbor_candidates = self._kdtree.query(
            self._kdtree.data, k=max_neighbors + 1, p=norm
        )

        # remove self reference
        distances = distances[:, 1:]
        neighbor_candidates = neighbor_candidates[:, 1:]

        min_distance = np.min(distances, axis=1)[:, np.newaxis]
        inside_max_distance = distances < (radius * min_distance)
        pixels, neigbor_index = np.nonzero(inside_max_distance)
        neighbors = neighbor_candidates[pixels, neigbor_index]
        data = np.ones(len(pixels), dtype=bool)

        neighbor_matrix = csr_matrix((data, (pixels, neighbors)))

        # filter annoying deprecation warning from within scipy
        # scipy still uses np.matrix in scipy.sparse, but we do not
        # explicitly use any feature of np.matrix, so we can ignore this here
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
            if (neighbor_matrix.T != neighbor_matrix).sum() > 0:
                warnings.warn(
                    "Neighbor matrix is not symmetric. Is camera geometry irregular?"
                )

        return neighbor_matrix

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

        return np.vstack(
            [
                x,
                y,
                x**2,
                x * y,
                y**2,
                x**3,
                x**2 * y,
                x * y**2,
                y**3,
                x**4,
                x**3 * y,
                x**2 * y**2,
                x * y**3,
                y**4,
            ]
        )

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

        angle: value convertible to an `astropy.coordinates.Angle`
            rotation angle with unit (e.g. 12 * u.deg), or "12d"

        """
        angle = Angle(angle)
        rotmat = rotation_matrix_2d(angle)
        rotated = np.dot(rotmat.T, [self.pix_x.value, self.pix_y.value])
        self.pix_x = rotated[0] * self.pix_x.unit
        self.pix_y = rotated[1] * self.pix_x.unit

        # do not use -=, copy is intentional here
        self.pix_rotation = self.pix_rotation - angle
        self.cam_rotation = Angle(0, unit=u.deg)

    def info(self, printer=print):
        """print detailed info about this camera"""
        printer(f'CameraGeometry: "{self}"')
        printer("   - num-pixels: {}".format(len(self.pix_id)))
        printer(f"   - pixel-type: {self.pix_type}")
        printer("   - sensitive-area: {}".format(self.pix_area.sum()))
        printer(f"   - pix-rotation: {self.pix_rotation}")
        printer(f"   - cam-rotation: {self.cam_rotation}")

    @classmethod
    def make_rectangular(
        cls, npix_x=40, npix_y=40, range_x=(-0.5, 0.5), range_y=(-0.5, 0.5)
    ):
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

        return cls(
            name="RectangularCamera",
            pix_id=ids,
            pix_x=xx,
            pix_y=yy,
            pix_area=(2 * rr) ** 2,
            neighbors=None,
            pix_type=PixelShape.SQUARE,
        )

    def get_border_pixel_mask(self, width=1):
        """
        Get a mask for pixels at the border of the camera of arbitrary width

        Parameters
        ----------
        width: int
            The width of the border in pixels

        Returns
        -------
        mask: array
            A boolean mask, True if pixel is in the border of the specified width
        """
        if width in self._border_cache:
            return self._border_cache[width]

        # filter annoying deprecation warning from within scipy
        # scipy still uses np.matrix in scipy.sparse, but we do not
        # explicitly use any feature of np.matrix, so we can ignore this here
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

            if width == 1:
                n_neighbors = np.asarray(self.neighbor_matrix_sparse.sum(axis=0))[0]
                max_neighbors = n_neighbors.max()
                mask = n_neighbors < max_neighbors
            else:
                n = self.neighbor_matrix
                mask = (n & self.get_border_pixel_mask(width - 1)).any(axis=1)

        self._border_cache[width] = mask
        return mask

    def position_to_pix_index(self, x, y):
        """
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
        """
        if not self._all_pixel_areas_equal:
            logger.warning(
                " Method not implemented for cameras with varying pixel sizes"
            )
        unit = x.unit
        scalar = x.ndim == 0

        points_searched = np.dstack([x.to_value(unit), y.to_value(unit)])
        circum_rad = self._pixel_circumradius[0].to_value(unit)
        kdtree = self._kdtree
        dist, pix_indices = kdtree.query(
            points_searched, distance_upper_bound=circum_rad
        )
        del dist
        pix_indices = pix_indices.flatten()

        invalid = np.iinfo(pix_indices.dtype).min
        # 1. Mark all points outside pixel circumeference as lying outside camera
        pix_indices[pix_indices == self.n_pixels] = invalid

        # 2. Accurate check for the remaining cases (within circumference, but still outside
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
                xprime = (
                    points_searched[0][index, 0]
                    - self.pix_x[borderpix_index].to_value(unit)
                    + self.pix_x[insidepix_index].to_value(unit)
                )
                yprime = (
                    points_searched[0][index, 1]
                    - self.pix_y[borderpix_index].to_value(unit)
                    + self.pix_y[insidepix_index].to_value(unit)
                )
                dist_check, index_check = kdtree.query(
                    [xprime, yprime], distance_upper_bound=circum_rad
                )
                del dist_check
                if index_check != insidepix_index:
                    pix_indices[index] = invalid

        return np.squeeze(pix_indices) if scalar else pix_indices

    @staticmethod
    def simtel_shape_to_type(pixel_shape):
        try:
            shape, rotation = SIMTEL_PIXEL_SHAPES[pixel_shape]
            # make sure we don't introduce a mutable global state
            return shape, rotation.copy()
        except KeyError:
            raise ValueError(f"Unknown pixel_shape {pixel_shape}") from None


class UnknownPixelShapeWarning(UserWarning):
    pass
