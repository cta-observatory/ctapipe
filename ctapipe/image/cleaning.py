"""
Image Cleaning Algorithms (identification of noisy pixels)
"""

__all__ = ['tailcuts_clean', 'dilate']

import numpy as np

def tailcuts_clean(geom, image, picture_thresh=7, boundary_thresh=5):
    """Clean an image by selection pixels that pass a two-threshold
    tail-cuts procedure.  The picture and boundary thresholds are
    defined with respect to the pedestal dispersion. All pixels that
    have a signal higher than the picture threshold will be retained,
    along with all those above the boundary threshold that are
    neighbors of a picture pixel.

    To include extra neighbor rows of pixels beyond what are accepted, use the
    `ctapipe.reco.dialate` function.

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

    Returns
    -------

    A boolean mask of *clean* pixels.  To get a zero-suppressed image and pixel
    list, use `image[mask], geom.pix_id[mask]`, or to keep the same
    image size and just set unclean pixels to 0 or similar, use
    `image[mask] = 0`

    """

    pixels_in_picture = image >= picture_thresh  # if pixel > p_thresh

    # make a 2d representation of all pixels that are in the picture
    # by simply repeating pixels_in_picture for each pixel,
    # so we can apply the neighbor matrix later with a simple multiplication
    npix = len(image)
    pixels_in_picture_2d = np.tile(pixels_in_picture, npix).reshape(npix, npix)

    # now find all pixels that are above the boundary threshold
    # AND have any neighbor that is in the picture
    pixels_above_boundary = image >= boundary_thresh
    pixels_with_picture_neighbors = (pixels_in_picture_2d
                                     * geom.neighbor_matrix).any(axis=1)

    return (pixels_above_boundary
            & pixels_with_picture_neighbors) | pixels_in_picture





def dilate(geom, mask):
    """Add one row of neighbors to the True values of a pixel mask.  This
    can be used to include extra rows of pixels in a mask that was
    pre-computed, e.g. via `tailcuts_clean`.

    Modifies mask in-place by default (pass `mask.copy()` if you want
    to maintain a copy of the undialated data)
    """
    for pixid in geom.pix_id[mask]:
        mask[geom.neighbors[pixid]] = True
