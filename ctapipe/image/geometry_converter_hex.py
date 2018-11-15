__author__ = "@tino-michael"

import logging

import numpy as np
from astropy import units as u
from numba import jit

from ctapipe.instrument import CameraGeometry

logger = logging.getLogger(__name__)

__all__ = [
    "convert_geometry_hex1d_to_rect2d",
    "convert_geometry_rect2d_back_to_hexe1d"
]


def unskew_hex_pixel_grid(pix_x, pix_y, cam_angle=0 * u.deg,
                          base_angle=60 * u.deg):
    r"""transform the pixel coordinates of a hexagonal image into an
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

    Notes
    -----
    The correction on the pixel position r can be described by a rotation R around
    one angle and a sheer S along a certain axis:

    .. math::
        r' = S \cdot R \cdot r

    .. math::
        \begin{pmatrix}
            x' \\
            y'
        \end{pmatrix}
        =
        \begin{pmatrix}
            1        &  0 \\
            -1/\tan  &  1
        \end{pmatrix}
        \cdot
        \begin{pmatrix}
            \cos  & -\sin \\
            \sin  &  \cos
        \end{pmatrix}
        \cdot
        \begin{pmatrix}
            x \\
            y
        \end{pmatrix}

    .. math::
        \begin{pmatrix}
            x' \\
            y'
        \end{pmatrix}
        =
        \begin{pmatrix}
                 \cos      &     -\sin      \\
            \sin-\cos/\tan & \sin/\tan+\cos
        \end{pmatrix}
        \cdot
        \begin{pmatrix}
            x \\
            y
        \end{pmatrix}

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
        # (x') = (   1    0) * (cos -sin) * (x) = (    cos         -sin    ) * (x)
        # (y')   (-1/tan  1)   (sin  cos)   (y)   (sin-cos/tan  sin/tan+cos) * (y)
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
    r"""skews the orthogonal coordinates back to the hexagonal ones

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

    Notes
    -----
    To revert the rotation, we need to find matrices S' and R' with
    :math:`S' \cdot S = 1` and :math:`R' \cdot R = 1`,
    so that :math:`r = R' \cdot S' \cdot S \cdot R \cdot r = R' \cdot S' \cdot  r'`:

    .. math::
        \begin{pmatrix}
            x \\
            y
        \end{pmatrix}
        =
        \begin{pmatrix}
            \cos  &  \sin \\
            -\sin &  \cos
        \end{pmatrix}
        \cdot
        \begin{pmatrix}
            1       &  0 \\
            1/\tan  &  1
        \end{pmatrix}
        \cdot
        \begin{pmatrix}
            x' \\
            y'
        \end{pmatrix}

    .. math::
        \begin{pmatrix}
            x \\
            y
        \end{pmatrix}
        =
        \begin{pmatrix}
            \cos+\sin/\tan  &  \sin \\
            \cos/\tan-\sin  &  \cos
        \end{pmatrix}
        \cdot
        \begin{pmatrix}
            x' \\
            y'
        \end{pmatrix}

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
        # r = R' * S' * S * R * r = R' * S' *  r'
        #
        # (x) = ( cos sin) * (  1    0) * (x') = (cos+sin/tan  sin) * (x')
        # (y)   (-sin cos)   (1/tan  1)   (y')   (cos/tan-sin  cos)   (y')

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
    d_x = 99 * u.meter  # TODO: @jit may have troubles interpreting astropy.Quantities
    d_y = 99 * u.meter
    x_base = pix_x[0]
    y_base = pix_y[0]
    for x, y in zip(pix_x, pix_y):
        if abs(y - y_base) < abs(x - x_base):
            d_x = min(d_x, abs(x - x_base))
        if abs(y - y_base) > abs(x - x_base):
            d_y = min(d_y, abs(y - y_base))

    # for x, y in zip(pix_x, pix_y):
    #    if abs(y - y_base) > abs(x - x_base):
    #        d_y = min(d_y, abs(y - y_base))

    x_scale = 1
    if scale_aspect:
        x_scale = d_y / d_x
        pix_x *= x_scale
        d_x = d_y

    # with the maximal extension of the axes and the size of the pixels,
    # determine the number of bins in each direction
    n_bins_x = (np.around(abs(max(pix_x) - min(pix_x)) / d_x) + 2).astype(int)
    n_bins_y = (np.around(abs(max(pix_y) - min(pix_y)) / d_y) + 2).astype(int)
    x_edges = np.linspace(min(pix_x).value, max(pix_x).value, n_bins_x)
    y_edges = np.linspace(min(pix_y).value, max(pix_y).value, n_bins_y)

    return x_edges, y_edges, x_scale


rot_buffer = {}


def convert_geometry_hex1d_to_rect2d(geom, signal, key=None, add_rot=0):
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
        arbitrary key (float, string) to store the transformed geometry in a buffer
        The geometries (hex and rect) will be stored in a buffer.
        The key is necessary to make the conversion back from rect to hex.
    add_rot : int/float (default: 0)
        parameter to apply an additional rotation of `add_rot` times 60°

    Returns
    -------
    new_geom : CameraGeometry object
        geometry object of the slanted picture now with a rectangular
        grid and a 2D grid for the pixel positions. contains now a 2D
        masking array signifying which of the pixels came from the
        original geometry and which are simply fillers from the
        rectangular grid
    rot_img : ndarray 2D (no timing) or 3D (with timing)
        the rectangular signal image

    Examples
    --------
    camera = event.inst.subarray.tel[tel_id].camera
    image = event.r0.tel[tel_id].image[0]
    key = camera.cam_id
    square_geom, square_image = convert_geometry_hex1d_to_rect2d(camera, image, key=key)
    """

    if key in rot_buffer:

        # if the conversion with this key was done before and stored,
        # just read it in
        (geom, new_geom, hex_to_rect_map) = rot_buffer[key]
    else:

        # otherwise, we have to do the conversion first now,
        # skew all the coordinates of the original geometry

        # extra_rot is the angle to get back to aligned hexagons with flat
        # tops. Note that the pixel rotation angle brings the camera so that
        # hexagons have a point at the top, so need to go 30deg back to
        # make them flat
        extra_rot = geom.pix_rotation - 30 * u.deg

        # total rotation angle:
        rot_angle = (add_rot * 60 * u.deg) - extra_rot

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
        pix_area = (np.ones_like(grid_x)
                    * (x_edges[1] - x_edges[0])
                    * (y_edges[1] - y_edges[0]))

        # creating a new geometry object with the attributes we just determined
        new_geom = CameraGeometry(
            cam_id=geom.cam_id + "_rect",
            pix_id=ids,  # this is a list of all the valid coordinate pairs now
            pix_x=grid_x * u.meter,
            pix_y=grid_y * u.meter,
            pix_area=pix_area * u.meter ** 2,
            neighbors=geom.neighbors,
            pix_type='rectangular', apply_derotation=False)

        # storing the pixel mask for later use
        new_geom.mask = square_mask

        # create a transfer map by enumerating all pixel positions in a 2D histogram
        hex_to_rect_map = np.histogramdd([rot_y, rot_x],
                                         bins=(y_edges, x_edges),
                                         weights=np.arange(len(signal)))[
            0].astype(int)
        # bins that do not correspond to the original image get an entry of `-1`
        hex_to_rect_map[~square_mask] = -1

        if signal.ndim > 1:
            long_map = []
            for i in range(signal.shape[-1]):
                tmp_map = hex_to_rect_map + i * (np.max(hex_to_rect_map) + 1)
                tmp_map[~square_mask] = -1
                long_map.append(tmp_map)
            hex_to_rect_map = np.array(long_map)

        if key is not None:
            # if a key is given, store the essential objects in a buffer
            rot_buffer[key] = (geom, new_geom, hex_to_rect_map)

    # done `if key in rot_buffer`

    # create the rotated rectangular image by applying `hex_to_rect_map` to the flat,
    # extended input image
    # `input_img_ext` is the flattened input image extended by one entry that contains NaN
    # since `hex_to_rect_map` contains `-1` for "fake" pixels, it maps this extra NaN
    # value at the last array position to any bin that does not correspond to a pixel of
    # the original image
    input_img_ext = np.full(np.prod(signal.shape) + 1, np.nan)

    # the way the map is produced, it has the time dimension as axis=0;
    # but `signal` has it as axis=-1, so we need to roll the axes back and forth a bit.
    # if there is no time dimension, `signal` is a 1d array and `rollaxis` has no effect.
    input_img_ext[:-1] = np.rollaxis(signal, axis=-1, start=0).ravel()

    # now apply the transfer map
    rot_img = input_img_ext[hex_to_rect_map]

    # if there is a time dimension, roll the time axis back to the last position
    try:
        rot_img = np.rollaxis(rot_img, 0, 3)
    except ValueError:
        pass

    return new_geom, rot_img


def convert_geometry_rect2d_back_to_hexe1d(geom, signal, key=None,
                                           add_rot=None):
    """reverts the geometry distortion performed by convert_geometry_hexe1d_to_rect_2d
    back to a hexagonal grid stored in 1D arrays

    Parameters
    ----------
    geom : CameraGeometry
        geometry object where pixel positions are stored in a 2D
        rectangular camera grid
    signal : ndarray
        pixel intensity stored in a 2D rectangular camera grid
    key:
        key to retrieve buffered geometry information
        (see `convert_geometry_hex1d_to_rect2d`)
    add_rot:
        not used -- only here for backwards compatibility

    Returns
    -------
    old_geom : CameraGeometry
        the original geometry of the image
    signal : ndarray
        1D (no timing) or 2D (with timing) array of the pmt signals

    Notes
    -----
    The back-conversion works with an internal buffer to store the transfer map (which
    was produced in the first conversion). If `key` is not found in said buffer, this
    function tries to perform a mock conversion. For this, it needs a `CameraGeometry`
    instance of the original camera layout, which it tries to load by name (i.e.
    the `cam_id`). The function assumes the original `cam_id` can be inferred from the
    given, modified one by: `geom.cam_id.split('_')[0]`.

    Examples
    --------
    camera = event.inst.subarray.tel[tel_id].camera
    image = event.r0.tel[tel_id].image[0]
    key = camera.cam_id
    square_geom, square_image = convert_geometry_hex1d_to_rect2d(camera, image, key=key)
    hex_geom, hex_image = convert_geometry_rect2d_back_to_hexe1d(square_geom,
    square_image, key = key)
    """

    if key not in rot_buffer:
        # if the key is not in the buffer from the initial conversion (maybe
        # because you did it in another process?), perform a mock conversion
        # here ATTENTION assumes the original cam_id can be inferred from the
        #  given, modified one by by `geom.cam_id.split('_')[0]`
        try:
            orig_geom = CameraGeometry.from_name(geom.cam_id.split('_')[0])
        except:
            raise ValueError(
                "could not deduce `CameraGeometry` from given `geom`...\n"
                "please provide a `geom`, so that "
                "`geom.cam_id.split('_')[0]` is a known `cam_id`")

        orig_signal = np.zeros(len(orig_geom.pix_x))
        convert_geometry_hex1d_to_rect2d(geom=orig_geom, signal=orig_signal,
                                         key=key, add_rot=add_rot)

    (old_geom, new_geom, hex_square_map) = rot_buffer[key]

    # the output image has as many entries as there are non-negative values in the
    # transfer map (this accounts for time as well)
    unrot_img = np.zeros(np.count_nonzero(hex_square_map >= 0))

    # rearrange input `signal` according to the mask and map
    # (the dots in the brackets expand the mask to account for a possible time dimension)
    # `atleast_3d` ensures that there is a third axis that we can roll to the front
    # even if there is no time; if we'd use `axis=-1` instead, in cas of no time
    # dimensions, we would rotate the x and y axes, resulting in a mirrored image
    # `squeeze` reduces the added axis again in the no-time-slices cases
    unrot_img[hex_square_map[..., new_geom.mask]] = \
        np.squeeze(np.rollaxis(np.atleast_3d(signal), 2, 0))[..., new_geom.mask]

    # if `signal` has a third dimension, that is the time
    # and we need to roll some axes again...
    if signal.ndim == 3:
        # unrot_img[hex_square_map[..., new_geom.mask]] = \
        # np.rollaxis(signal, -1, 0)[..., new_geom.mask]

        # reshape the image so that the time is the first axis
        # and then roll the time to the back
        unrot_img = unrot_img.reshape((signal.shape[2],
                                       np.count_nonzero(new_geom.mask)))
        unrot_img = np.rollaxis(unrot_img, -1, 0)
    # else:
    #     unrot_img[hex_square_map[new_geom.mask]] = \
    #         signal[new_geom.mask]

    return old_geom, unrot_img
