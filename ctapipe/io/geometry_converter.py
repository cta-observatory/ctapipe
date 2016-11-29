import numpy as np

from astropy import units as u

from numba import jit

from .camera import CameraGeometry

from ctapipe.utils.linalg import rotation_matrix_2d

def unskew_hex_pixel_grid(pix_x, pix_y, cam_angle=0*u.deg, base_angle=60*u.deg):
    """
        transform the pixel coordinates of a hexagonal image into an orthogonal image

        Parameters:
        -----------
        pix_x, pix_y : 1D numpy arrays
            the list of x and y coordinates of the hexagonal pixel grid
        cam_angle : astropy.Quantity (default: 0 degrees)
            some cameras have a weird rotation of their grid
            this needs to be corrected since the skewing is performed along the y-axis
            therefore one of the slanted base-vectors needs to be parallel to the y-axis
        base_angle : astropy.Quantity (default: 60 degrees)
            the skewing angle of the hex-grid. should be 60° for regular hexagons

        Returns:
        --------
        pix_x, pix_y : 1D numpy arrays
            the list of x and y coordinates of the slanted, orthogonal pixel grid
    """

    tan_angle = np.tan(base_angle)

    '''
    if the camera grid is rotated as a whole, we have to undo that too: '''
    if cam_angle != 0*u.deg:
        sin_angle = np.sin(cam_angle-270*u.deg)
        cos_angle = np.cos(cam_angle-270*u.deg)

        '''
        the correction on the pixel position r can be described by a rotation R around one
        angle and a sheer S along a certain axis:

         r'  = S * R * r
        (x') = (1       0) * (cos -sin) * (x) = (cos -sin        -sin      ) * (x)
        (y')   (-1/tan  1)   (sin  cos)   (y)   (sin-cos/tan  sin/tan + cos)   (y)
        TODO put that in latex...
        '''

        rot_mat = np.array(
            [[cos_angle, -sin_angle],
             [sin_angle-cos_angle/tan_angle, sin_angle/tan_angle+cos_angle]])

    else:
        ''' if we don't rotate the camera, only perform the sheer '''
        rot_mat = np.array([[1, 0], [-1/tan_angle, 1]])

    rotated = np.dot(rot_mat, [pix_x.value, pix_y.value])
    rot_x = rotated[0] * pix_x.unit
    rot_y = rotated[1] * pix_x.unit
    return rot_x, rot_y


def reskew_hex_pixel_grid(pix_x, pix_y, cam_angle=0*u.deg, base_angle=60*u.deg):
    """
        skews the orthogonal coordinates back to the hexagonal ones

        Parameters:
        -----------
        pix_x, pix_y : 1D numpy arrays
            the list of x and y coordinates of the slanted, orthogonal pixel grid
        cam_angle : astropy.Quantity (default: 0 degrees)
            some cameras have a weird rotation of their grid
            this needs to be corrected since the skewing is performed along the y-axis
            therefore one of the slanted base-vectors needs to be parallel to the y-axis
        base_angle : astropy.Quantity (default: 60 degrees)
            the skewing angle of the hex-grid. should be 60° for regular hexagons

        Returns:
        --------
        pix_x, pix_y : 1D numpy arrays
            the list of x and y coordinates of the hexagonal pixel grid
    """

    tan_angle = np.tan(base_angle)

    '''
    if the camera grid is rotated as a whole, we have to undo that, too
    the -270 degrees are chosen so that the rotated/slanted image has about the same
    orientation as the original
    Note: since CameraGeometry.guess(pix_x, pix_y, optical_foclen) rotates the pixel image
    itself, we are not going to do that here and only apply the -270 degrees rotation '''
    if cam_angle != 0*u.deg:
        sin_angle = np.sin(-270*u.deg)
        cos_angle = np.cos(-270*u.deg)

        '''
        to revert the rotation, we need to find matrices S' and R'
        S' * S = 1 and R' * R = 1
        so that
        r = R' * S' * S * R * r = R' * S'*  r'

        (x) = ( cos sin) * (1      0) * (x') = (cos+sin/tan  sin) * (x)
        (y)   (-sin cos)   (1/tan  1)   (y')   (cos/tan-sin  cos)   (y)
        TODO put that in latex...
        '''

        rot_mat = np.array(
            [[cos_angle+sin_angle/tan_angle, sin_angle],
             [cos_angle/tan_angle-sin_angle, cos_angle]])

    else:
        ''' if we don't rotate the camera, only perform the sheer '''
        rot_mat = np.array([[1, 0], [1/tan_angle, 1]])

    rotated = np.dot(rot_mat, [pix_x.value, pix_y.value])
    rot_x = rotated[0] * pix_x.unit
    rot_y = rotated[1] * pix_x.unit
    return rot_x, rot_y


@jit
def reskew_hex_pixel_from_orthogonal_edges(x_edges, y_edges, square_mask):
    """
        extracts and skews the pixel coordinates from a 2D orthogonal histogram
        (i.e. the bin-edges) and skews them into the hexagonal image while selecting only
        the pixel that are selected by the given mask

        Parameters:
        -----------
        x_edges, y_edges : 1darrays
            the bin edges of the 2D histogram
        square_mask : 2darray
            mask that selects the pixels actually belonging to the camera

        Returns:
        --------
        unrot_x, unrot_y : 1darrays
            pixel coordinated reskewed into the hexagonal camera grid
    """

    unrot_x, unrot_y = [], []
    for i, x in enumerate((x_edges[:-1]+x_edges[1:])/2):
        for j, y in enumerate((y_edges[:-1]+y_edges[1:])/2):
            if square_mask[i][j]:
                x_unrot, y_unrot = reskew_hex_pixel_grid(x, y)
                unrot_x.append(x_unrot)
                unrot_y.append(y_unrot)
    return unrot_x, unrot_y


@jit
def get_orthogonal_grid_edges(pix_x, pix_y, scale_aspect=True):
    """
        calculate the bin edges of the slanted, orthogonal pixel grid to resample the
        pixel signals with np.histogramdd right after.

        Parameters:
        -----------
        pix_x, pix_y : 1D numpy arrays
            the list of x and y coordinates of the slanted, orthogonal pixel grid

        scale_aspect : boolean (default: True)
            if True, rescales the x-coordinates to create square pixels (instead of
            rectangular ones)

        Returns:
        --------
        x_edges, y_edges : 1D numpy arrays
            the bin edges for the slanted, orthogonal pixel grid
        x_scale : float
            factor by which the x-coordinates have been scaled
    """

    '''
    finding the size of the square patches '''
    d_x = 99 * u.m
    d_y = 99 * u.m
    x_base = pix_x[0]
    y_base = pix_y[0]
    for x, y in zip(pix_x, pix_y):
        if abs(y-y_base) < abs(x-x_base):
            d_x = min(d_x, abs(x-x_base))
    for x, y in zip(pix_x, pix_y):
        if abs(y-y_base) > abs(x-x_base):
            d_y = min(d_y, abs(y-y_base))

    x_scale = 1
    if scale_aspect:
        x_scale = d_y / d_x
        pix_x *= x_scale
        d_x = d_y

    '''
    with the maximal extension of the axes and the size of the pixels, determine the
    number of bins in each direction '''
    NBinsx = np.around(abs(max(pix_x) - min(pix_x))/d_x) + 2
    NBinsy = np.around(abs(max(pix_y) - min(pix_y))/d_y) + 2
    x_edges = np.linspace(min(pix_x).value, max(pix_x).value, NBinsx)
    y_edges = np.linspace(min(pix_y).value, max(pix_y).value, NBinsy)

    return x_edges, y_edges, x_scale


rot_buffer = {}
def convert_geometry_1d_to_2d(geom, signal, key=None):

    if key in rot_buffer:
        (rot_x, rot_y, x_edges, y_edges, square_mask, x_scale) = rot_buffer[key]
    else:
        rot_x, rot_y = unskew_hex_pixel_grid(geom.pix_x, geom.pix_y,
                                             geom.cam_rotation)

        x_edges, y_edges, x_scale = get_orthogonal_grid_edges(rot_x, rot_y)

        square_mask = np.histogramdd([rot_x, rot_y],
                                     bins=(x_edges, y_edges))[0]

        if key is not None:
            rot_buffer[key] = (rot_x, rot_y, x_edges, y_edges, square_mask, x_scale)

    square_img = np.histogramdd([rot_x, rot_y],
                                bins=(x_edges, y_edges),
                                weights=signal)[0]

    ids = []
    for i, row in enumerate(square_mask):
        for j, val in enumerate(row):
            if val is True:
                ids.append((i, j))

    grid_x, grid_y = np.meshgrid((x_edges[:-1] + x_edges[1:])/2.,
                                 (y_edges[:-1] + y_edges[1:])/2.)

    pix_area = np.ones_like(grid_x) \
        * (x_edges[1]-x_edges[0]) * (y_edges[1]-y_edges[0])

    new_geom = CameraGeometry(
        cam_id=geom.cam_id,
        pix_id=ids,  # this is a list of all the valid coordinate pairs now
        pix_x=grid_x * u.m,
        pix_y=grid_y * u.m,
        pix_area=pix_area * u.m**2,
        neighbors=None,  # TODO? ... it's a 2D grid after all ...
        pix_type='rectangular')

    new_geom.mask = square_mask.T
    new_geom.cam_rotation = geom.cam_rotation

    return new_geom, square_img.T


def convert_geometry_back(geom, signal, key, foc_len):

    if key in rot_buffer:
        x_scale = rot_buffer[key][-1]
    else:
        raise Exception("key '{}' not found in the buffer".format(key))

    grid_x, grid_y = geom.pix_x / x_scale, geom.pix_y
    square_mask = geom.mask

    unrot_x, unrot_y = reskew_hex_pixel_grid(grid_x[square_mask == 1],
                                             grid_y[square_mask == 1],
                                             geom.cam_rotation)

    unrot_geom = CameraGeometry.guess(unrot_x, unrot_y, foc_len)

    return unrot_geom, signal[square_mask == 1]
