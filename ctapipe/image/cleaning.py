"""
Image Cleaning Algorithms (identification of noisy pixels)
"""

__all__ = ['tailcuts_clean', 'dilate']


def tailcuts_clean(geom, image, pedvars, picture_thresh=4.25,
                   boundary_thresh=2.25):
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
    geom: `ctapipe.io.CameraGeometry`
        Camera geometry information
    image: array
        pedestal-subtracted, flat-fielded pixel values
    pedvars: array or scalar
        pedestal dispersion of all pixels, or any other
        multiplicative factor that one wants to use to normalize the
        thresholds (e.g. if your image is already in PE units, this could
        simply be set to 1, and the thresholds defined in PE)
    picture_thresh: float
        high threshold as multiple of the pedvar
    boundary_thresh: float
        low-threshold as mutiple of pedvar (+ nearest neighbor)

    Returns
    -------

    A boolean mask of *clean* pixels.  To get a zero-suppressed image and pixel
    list, use `image[mask], geom.pix_id[mask]`, or to keep the same
    image size and just set unclean pixels to 0 or similar, use
    `image[mask] = 0`

    """

    clean_mask = image >= picture_thresh * pedvars  # starts as picture pixels

    # good boundary pixels are those that have any picture pixel as a
    # neighbor
    boundary_mask = image >= boundary_thresh * pedvars
    boundary_ids = [pix_id for pix_id in geom.pix_id[boundary_mask]
                    if clean_mask[geom.neighbors[pix_id]].any()]

    clean_mask[boundary_ids] = True
    return clean_mask


def dilate(geom, mask):
    """Add one row of neighbors to the True values of a pixel mask.  This
    can be used to include extra rows of pixels in a mask that was
    pre-computed, e.g. via `tailcuts_clean`.

    Modifies mask in-place by default (pass `mask.copy()` if you want
    to maintain a copy of the undialated data)
    """
    for pixid in geom.pix_id[mask]:
        mask[geom.neighbors[pixid]] = True
