import astropy.units as u
import numpy as np

__all__ = [
    "unskew_hex_pixel_grid",
    "reskew_hex_pixel_grid",
    "reskew_hex_pixel_from_orthogonal_edges",
    "get_orthogonal_grid_edges",
    "get_orthogonal_grid_indices",
]


def get_orthogonal_grid_indices(pos, size):
    """
    Bin pixel positions on a square grid with bin widths at least half the pixel size.
    This can be used to infer the rows and columns.
    """
    rnd = np.round((pos / size).to_value(u.dimensionless_unscaled), 1)
    unique = np.sort(np.unique(rnd))
    mask = np.append(np.diff(unique) > 0.5, True)
    bins = np.append(unique[mask] - 0.5, unique[-1] + 0.5)
    return np.digitize(rnd, bins) - 1


def unskew_hex_pixel_grid(pix_x, pix_y, cam_angle=0 * u.deg, base_angle=60 * u.deg):
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
            [
                [cos_angle, -sin_angle],
                [sin_angle - cos_angle / tan_angle, sin_angle / tan_angle + cos_angle],
            ]
        )

    else:
        # if we don't rotate the camera, only perform the sheer
        rot_mat = np.array([[1, 0], [-1 / tan_angle, 1]])

    rotated = np.dot(rot_mat, [pix_x.value, pix_y.value])
    rot_x = rotated[0] * pix_x.unit
    rot_y = rotated[1] * pix_x.unit
    return rot_x, rot_y


def reskew_hex_pixel_grid(pix_x, pix_y, cam_angle=0 * u.deg, base_angle=60 * u.deg):
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
            [
                [cos_angle + sin_angle / tan_angle, sin_angle],
                [cos_angle / tan_angle - sin_angle, cos_angle],
            ]
        )

    else:
        # if we don't rotate the camera, only perform the sheer
        rot_mat = np.array([[1, 0], [1 / tan_angle, 1]])

    rotated = np.dot(rot_mat, [pix_x.value, pix_y.value])
    rot_x = rotated[0] * pix_x.unit
    rot_y = rotated[1] * pix_x.unit
    return rot_x, rot_y


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


def get_orthogonal_grid_edges(pix_x, pix_y, scale_aspect=True):
    """calculate the bin edges of the slanted, orthogonal pixel grid to
    resample the pixel signals with np.histogramdd right after.

    Parameters
    ----------
    pix_x, pix_y : 1D numpy arrays
        the list of x and y coordinates of the slanted, orthogonal pixel grid
        units should be in meters, and stripped off
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

    d_x = np.inf
    d_y = np.inf
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
    n_bins_x = int(np.round(np.abs(np.max(pix_x) - np.min(pix_x)) / d_x) + 2)
    n_bins_y = int(np.round(np.abs(np.max(pix_y) - np.min(pix_y)) / d_y) + 2)
    x_edges = np.linspace(pix_x.min(), pix_x.max(), n_bins_x)
    y_edges = np.linspace(pix_y.min(), pix_y.max(), n_bins_y)

    return (x_edges, y_edges, x_scale)
