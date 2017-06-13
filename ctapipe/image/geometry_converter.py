import logging
from collections import namedtuple
import numpy as np
from astropy import units as u
from numba import jit

from ctapipe.instrument import CameraGeometry

logger = logging.getLogger(__name__)

RotBuffer = namedtuple("RotBuffer",
                       "rot_x,rot_y,x_edges,y_edges,new_geom,rot_angle,pix_rotation,"
                       "x_scale")


def unskew_hex_pixel_grid(pix_x, pix_y, cam_angle=0 * u.deg,
                          base_angle=60 * u.deg):
    """transform the pixel coordinates of a hexagonal image into an
    orthogonal image

    Parameters
    ----------
    pix_x, pix_y : 1D numpy arrays
        the list of x and y coordinates of the hexagonal pixel grid
    cam_angle : astropy.Quantity (default: 0 degrees)
        The skewing is performed along the y-axis, therefore, one of the slanted
        base-vectors needs to be parallel to the y-axis.
        Some camera grids are rotated in a way that this is not the case.
        This needs to be corrected.
    base_angle : astropy.Quantity (default: 60 degrees)
        the skewing angle of the hex-grid. should be 60° for regular hexagons

    Returns
    -------
    pix_x, pix_y : 1D numpy arrays
        the list of x and y coordinates of the slanted, orthogonal pixel grid
    """

    tan_angle = np.tan(base_angle)

    # If the camera-rotation angle is non-zero, create a rotation+sheering
    # matrix for the pixel coordinates
    if cam_angle != 0 * u.deg:
        sin_angle = np.sin(cam_angle)
        cos_angle = np.cos(cam_angle)

        # the correction on the pixel position r can be described by a
        # rotation R around one angle and a sheer S along a certain axis:
        #
        #  r'  = S * R * r
        # (x') = (   1    0) * (cos -sin) * (x) = (    cos         -sin    )
        # * (x)
        # (y')   (-1/tan  1)   (sin  cos)   (y)   (sin-cos/tan  sin/tan+cos)
        #   (y)
        # TODO put that in latex...

        rot_mat = np.array(
            [[cos_angle, -sin_angle],
             [sin_angle - cos_angle / tan_angle,
              sin_angle / tan_angle + cos_angle]])

    else:
        # if we don't rotate the camera, only perform the sheer
        rot_mat = np.array([[1, 0], [-1 / tan_angle, 1]])

    rotated = np.dot(rot_mat, [pix_x.value, pix_y.value])
    rot_x = rotated[0] * pix_x.unit
    rot_y = rotated[1] * pix_x.unit
    return rot_x, rot_y


def reskew_hex_pixel_grid(pix_x, pix_y, cam_angle=0 * u.deg,
                          base_angle=60 * u.deg):
    """skews the orthogonal coordinates back to the hexagonal ones

    Parameters
    ----------
    pix_x, pix_y : 1D numpy arrays
        the list of x and y coordinates of the slanted, orthogonal pixel grid
    cam_angle : astropy.Quantity (default: 0 degrees)
        The skewing is performed along the y-axis, therefore, one of the slanted
        base-vectors needs to be parallel to the y-axis.
        Some camera grids are rotated in a way that this is not the case.
        This needs to be corrected.
    base_angle : astropy.Quantity (default: 60 degrees)
        the skewing angle of the hex-grid. should be 60° for regular hexagons

    Returns
    -------
    pix_x, pix_y : 1D numpy arrays
        the list of x and y coordinates of the hexagonal pixel grid

    """

    tan_angle = np.tan(base_angle)

    # If the camera-rotation angle is non-zero, create a rotation+sheering
    # matrix for the pixel coordinates
    if cam_angle != 0 * u.deg:
        sin_angle = np.sin(cam_angle)
        cos_angle = np.cos(cam_angle)

        # to revert the rotation, we need to find matrices S' and R'
        # S' * S = 1 and R' * R = 1
        # so that
        # r = R' * S' * S * R * r = R' * S'*  r'
        #
        # (x) = ( cos sin) * (  1    0) * (x') = (cos+sin/tan  sin) * (x')
        # (y)   (-sin cos)   (1/tan  1)   (y')   (cos/tan-sin  cos)   (y')
        # TODO put that in latex...

        rot_mat = np.array(
            [[cos_angle + sin_angle / tan_angle, sin_angle],
             [cos_angle / tan_angle - sin_angle, cos_angle]])

    else:
        # if we don't rotate the camera, only perform the sheer
        rot_mat = np.array([[1, 0], [1 / tan_angle, 1]])

    rotated = np.dot(rot_mat, [pix_x.value, pix_y.value])
    rot_x = rotated[0] * pix_x.unit
    rot_y = rotated[1] * pix_x.unit
    return rot_x, rot_y


@jit
def reskew_hex_pixel_from_orthogonal_edges(x_edges, y_edges, square_mask):
    """extracts and skews the pixel coordinates from a 2D orthogonal
    histogram (i.e. the bin-edges) and skews them into the hexagonal
    image while selecting only the pixel that are selected by the
    given mask

    Parameters
    ----------
    x_edges, y_edges : 1darrays
        the bin edges of the 2D histogram
    square_mask : 2darray
        mask that selects the pixels actually belonging to the camera

    Returns
    -------
    unrot_x, unrot_y : 1darrays
        pixel coordinated reskewed into the hexagonal camera grid
    """

    unrot_x, unrot_y = [], []
    for i, x in enumerate((x_edges[:-1] + x_edges[1:]) / 2):
        for j, y in enumerate((y_edges[:-1] + y_edges[1:]) / 2):
            if square_mask[i][j]:
                x_unrot, y_unrot = reskew_hex_pixel_grid(x, y)
                unrot_x.append(x_unrot)
                unrot_y.append(y_unrot)
    return unrot_x, unrot_y


@jit
def get_orthogonal_grid_edges(pix_x, pix_y, scale_aspect=True):
    """calculate the bin edges of the slanted, orthogonal pixel grid to
    resample the pixel signals with np.histogramdd right after.

    Parameters
    ----------
    pix_x, pix_y : 1D numpy arrays
        the list of x and y coordinates of the slanted, orthogonal pixel grid
    scale_aspect : boolean (default: True)
        if True, rescales the x-coordinates to create square pixels
        (instead of rectangular ones)

    Returns
    --------
    x_edges, y_edges : 1D numpy arrays
        the bin edges for the slanted, orthogonal pixel grid
    x_scale : float
        factor by which the x-coordinates have been scaled
    """

    # finding the size of the square patches
    d_x = 99 * u.m
    d_y = 99 * u.m
    x_base = pix_x[0]
    y_base = pix_y[0]
    for x, y in zip(pix_x, pix_y):
        if abs(y - y_base) < abs(x - x_base):
            d_x = min(d_x, abs(x - x_base))
    for x, y in zip(pix_x, pix_y):
        if abs(y - y_base) > abs(x - x_base):
            d_y = min(d_y, abs(y - y_base))

    x_scale = 1
    if scale_aspect:
        x_scale = d_y / d_x
        pix_x *= x_scale
        d_x = d_y

    # with the maximal extension of the axes and the size of the pixels,
    # determine the number of bins in each direction
    NBinsx = np.around(abs(max(pix_x) - min(pix_x)) / d_x) + 2
    NBinsy = np.around(abs(max(pix_y) - min(pix_y)) / d_y) + 2
    x_edges = np.linspace(min(pix_x).value, max(pix_x).value, NBinsx)
    y_edges = np.linspace(min(pix_y).value, max(pix_y).value, NBinsy)

    return x_edges, y_edges, x_scale


add_angle = 180 * u.deg
rot_buffer = {}


def convert_geometry_1d_to_2d(geom, signal, key=None, add_rot=0):
    """converts the geometry object of a camera with a hexagonal grid into
    a square grid by slanting and stretching the 1D arrays of pixel x
    and y positions and signal intensities are converted to 2D
    arrays. If the signal array contains a time-dimension it is
    conserved.

    Parameters
    ----------
    geom : CameraGeometry object
        geometry object of hexagonal cameras
    signal : ndarray
        1D (no timing) or 2D (with timing) array of the pmt signals
    key : (default: None)
        arbitrary key to store the transformed geometry in a buffer
    add_rot : int/float (default: 0)
        parameter to apply an additional rotation of @add_rot times 60°

    Returns
    -------
    new_geom : CameraGeometry object
        geometry object of the slanted picture now with a rectangular
        grid and a 2D grid for the pixel positions contains now a 2D
        masking array signifying which of the pixels came from the
        original geometry and which are simply fillers from the
        rectangular grid square_img : ndarray 2D (no timing) or 3D
        (with timing) array of the pmt signals

    """

    if key in rot_buffer:

        # if the conversion with this key was done and stored before,
        # just read it in
        (rot_x, rot_y, x_edges, y_edges, new_geom,
         rot_angle, pix_rotation, x_scale) = rot_buffer[key]
    else:

        # otherwise, we have to do the conversion now first, skew all
        # the coordinates of the original geometry

        # extra_rot is the angle to get back to aligned hexagons with flat
        # tops. Note that the pixel rotation angle brings the camera so that
        # hexagons have a point at the top, so need to go 30deg back to
        # make them flat
        extra_rot = geom.pix_rotation - 30*u.deg

        # total rotation angle:
        rot_angle = (add_rot * 60 * u.deg) - extra_rot
        # if geom.cam_id.startswith("NectarCam")\
        #         or geom.cam_id.startswith("LSTCam"):
        #     rot_angle += geom.cam_rotation + 90 * u.deg

        logger.debug("geom={}".format(geom))
        logger.debug("rot={}, extra={}".format(rot_angle, extra_rot))

        rot_x, rot_y = unskew_hex_pixel_grid(geom.pix_x, geom.pix_y,
                                             cam_angle=rot_angle)

        # with all the coordinate points, we can define the bin edges
        # of a 2D histogram
        x_edges, y_edges, x_scale = get_orthogonal_grid_edges(rot_x, rot_y)

        # this histogram will introduce bins that do not correspond to
        # any pixel from the original geometry. so we create a mask to
        # remember the true camera pixels by simply throwing all pixel
        # positions into numpy.histogramdd: proper pixels contain the
        # value 1, false pixels the value 0.
        square_mask = np.histogramdd([rot_y, rot_x],
                                     bins=(y_edges, x_edges))[0].astype(bool)

        # to be consistent with the pixel intensity, instead of saving
        # only the rotated positions of the true pixels (rot_x and
        # rot_y), create 2D arrays of all x and y positions (also the
        # false ones).
        grid_x, grid_y = np.meshgrid((x_edges[:-1] + x_edges[1:]) / 2.,
                                     (y_edges[:-1] + y_edges[1:]) / 2.)

        ids = []
        # instead of blindly enumerating all pixels, let's instead
        # store a list of all valid -- i.e. picked by the mask -- 2D
        # indices
        for i, row in enumerate(square_mask):
            for j, val in enumerate(row):
                if val is True:
                    ids.append((i, j))

        # the area of the pixels (note that this is still a deformed
        # image)
        pix_area = np.ones_like(grid_x) \
                   * (x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0])

        # creating a new geometry object with the attributes we just determined
        new_geom = CameraGeometry(
            cam_id=geom.cam_id+"_rect",
            pix_id=ids,  # this is a list of all the valid coordinate pairs now
            pix_x=grid_x * u.m,
            pix_y=grid_y * u.m,
            pix_area=pix_area * u.m ** 2,
            neighbors=geom.neighbors,
            pix_type='rectangular', apply_derotation=False)

        # storing the pixel mask and camera rotation for later use
        new_geom.mask = square_mask

        if key is not None:
            # if a key is given, store the essential objects in a buffer
            rot_buffer[key] = RotBuffer(rot_x, rot_y, x_edges, y_edges, new_geom,
                                        rot_angle, geom.pix_rotation, x_scale)

    # resample the signal array to correspond to the square grid --
    #  for signal arrays containing time slices (ndim > 1) or not
    #  approach is the same as used for the mask only with the signal
    #  as bin-weights
    if signal.ndim > 1:
        t_dim = signal.shape[1]
        square_img = np.histogramdd([np.repeat(rot_y, t_dim),
                                     np.repeat(rot_x, t_dim),
                                     [a for a in range(t_dim)] * len(rot_x)],
                                    bins=(y_edges, x_edges, range(t_dim + 1)),
                                    weights=signal.ravel())[0]
    else:
        square_img = np.histogramdd([rot_y, rot_x],
                                    bins=(y_edges, x_edges),
                                    weights=signal)[0]

    return new_geom, square_img


unrot_buffer = {}  # todo: should just make this a function with @lru_cache


def convert_geometry_back(geom, signal, key, add_rot=0):
    """reverts the geometry distortion performed by
    convert_geometry_1d_to_2d back to a hexagonal grid stored in 1D
    arrays

    Parameters
    ----------
    geom : CameraGeometry
        geometry object where pixel positions are stored in a 2D
        rectangular camera grid
    signal : ndarray
        pixel intensity stored in a 2D rectangular camera grid
    key:
        key to retrieve buffered geometry information
    add_rot : int/float (default: 0)
        parameter to apply an additional rotation of @add_rot times 60°

    Returns
    -------
    unrot_geom : CameraGeometry
        pixel rotated back to a hexagonal grid stored in a 1D array
    signal : ndarray
        1D (no timing) or 2D (with timing) array of the pmt signals

    """
    global unrot_buffer

    square_mask = geom.mask

    if key in unrot_buffer:
        unrot_geom = unrot_buffer[key]
    else:
        if key in rot_buffer:
            x_scale = rot_buffer[key].x_scale
            rot_angle = rot_buffer[key].rot_angle
            pix_rotation = rot_buffer[key].pix_rotation
        else:
            raise KeyError("key '{}' not found in the buffer".format(key))

        grid_x, grid_y = geom.pix_x / x_scale, geom.pix_y

        unrot_x, unrot_y = reskew_hex_pixel_grid(grid_x[square_mask],
                                                 grid_y[square_mask],
                                                 rot_angle)

        # TODO: probably should use base constructor, not guess here:
        # unrot_geom = CameraGeometry.guess(unrot_x, unrot_y, foc_len,
        #                                  apply_derotation=False)
        unrot_geom = CameraGeometry(cam_id=geom.cam_id + "_hex",
                                    pix_id=np.arange(len(unrot_x)),
                                    pix_x=unrot_x,
                                    pix_y=unrot_y,
                                    pix_area=None,  # recalc
                                    pix_type='hexagonal',
                                    pix_rotation=pix_rotation,
                                    # apply_derotation=False
                                    )

        unrot_buffer[key] = unrot_geom

    return unrot_geom, signal[square_mask, ...]
