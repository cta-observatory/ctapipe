"""
Image Cleaning Algorithms (identification of noisy pixels)
"""

__all__ = ['tailcuts_clean', 'dilate']

import numpy as np


def tailcuts_clean(geom, image, picture_thresh=7, boundary_thresh=5,
                   keep_isolated_pixels=False,
                   min_number_picture_neighbors=0):

    """Clean an image by selection pixels that pass a two-threshold
    tail-cuts procedure.  The picture and boundary thresholds are
    defined with respect to the pedestal dispersion. All pixels that
    have a signal higher than the picture threshold will be retained,
    along with all those above the boundary threshold that are
    neighbors of a picture pixel.

    To include extra neighbor rows of pixels beyond what are accepted, use the
    `ctapipe.image.dilate` function.

    Parameters
    ----------
    geom: `ctapipe.instrument.CameraGeometry`
        Camera geometry information
    image: array
        pixel values
    picture_thresh: float or array
        threshold above which all pixels are retained
    boundary_thresh: float or array
        threshold above which pixels are retained if they have a neighbor 
        already above the picture_thresh
    keep_isolated_pixels: bool
        If True, pixels above the picture threshold will be included always, 
        if not they are only included if a neighbor is in the picture or 
        boundary
    min_number_picture_neighbors: int
        A picture pixel survives cleaning only if it has at least this number
        of picture neighbors. This has no effect in case keep_isolated_pixels is True

    Returns
    -------

    A boolean mask of *clean* pixels.  To get a zero-suppressed image and pixel
    list, use `image[mask], geom.pix_id[mask]`, or to keep the same
    image size and just set unclean pixels to 0 or similar, use
    `image[~mask] = 0`

    """
    pixels_above_picture = image >= picture_thresh

    if keep_isolated_pixels or min_number_picture_neighbors == 0:
        pixels_in_picture = pixels_above_picture
    else:
        # Require at least min_number_picture_neighbors. Otherwise, the pixel
        #  is not selected
        number_of_neighbors_above_picture = np.sum(pixels_above_picture &
                                                   geom.neighbor_matrix, axis=1)
        pixels_in_picture = pixels_above_picture & (
            number_of_neighbors_above_picture >= min_number_picture_neighbors
        )

    # by broadcasting together pixels_in_picture (1d) with the neighbor
    # matrix (2d), we find all pixels that are above the boundary threshold
    # AND have any neighbor that is in the picture
    pixels_above_boundary = image >= boundary_thresh
    pixels_with_picture_neighbors = (pixels_in_picture &
                                     geom.neighbor_matrix).any(axis=1)

    if keep_isolated_pixels:
        return (pixels_above_boundary
                & pixels_with_picture_neighbors) | pixels_in_picture
    else:
        pixels_with_boundary_neighbors = (pixels_above_boundary &
                                          geom.neighbor_matrix).any(axis=1)

        return ((pixels_above_boundary & pixels_with_picture_neighbors) |
                (pixels_in_picture & pixels_with_boundary_neighbors))


def dilate(geom, mask):
    """
    Add one row of neighbors to the True values of a pixel mask and return 
    the new mask.
    This can be used to include extra rows of pixels in a mask that was
    pre-computed, e.g. via `tailcuts_clean`.

    Parameters
    ----------
    geom: `~ctapipe.instrument.CameraGeometry`
        Camera geometry information
    mask: ndarray 
        input mask (array of booleans) to be dilated
    """
    return mask | (mask & geom.neighbor_matrix).any(axis=1)
