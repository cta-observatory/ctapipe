"""
Image Cleaning Algorithms (identification of noisy pixels)
"""

__all__ = ['tailcuts_clean']

def tailcuts_clean(geom, image, pedvars, picture_thresh=4.25,
                   boundary_thresh=2.25):
    """Clean an image by selection pixels that pass a two-threshold
    tail-cuts procedure.  The picture and boundary thresholds are
    defined with respect to the pedestal dispersion. All pixels that
    have a signal higher than the picture threshold will be retained,
    along with all those above the boundary threshold that are
    neighbors of a picture pixel.

    Parameters
    ----------
    geom: `CameraGeometry`
        Camera geometry information
    image: array
        pedestal-subtracted, flat-fielded pixel values
    pedvars: array
        pedestal dispersion corresponding to image
    picture_thresh: float
        high threshold as multiple of the pedvar
    boundary_thresh: float
        low-threshold as mutiple of pedvar (+ nearest neighbor)

    Returns:
    --------

    A boolean mask of "clean" pixels (to get a clean image just use
    `image[mask]`, or to get their pixel ids use `geom.pix_id[mask]`
    """

    clean_mask = image >= picture_thresh * pedvars  # starts as picture pixels

    # good boundary pixels are those that have any picture pixel as a
    # neighbor
    boundary_mask = image >= boundary_thresh * pedvars
    boundary_ids = [pix_id for pix_id in geom.pix_id[boundary_mask]
                    if clean_mask[geom.neighbors[pix_id]].any()]

    clean_mask[boundary_ids] = True
    return clean_mask


if __name__ == '__main__':

    from ctapipe import io
    import numpy as np

    geom = io.CameraGeometry.from_name("HESS", 1)
    image = np.zeros_like(geom.pix_id, dtype=np.float)
    pedvar = np.ones_like(geom.pix_id, dtype=np.float)

    # some test data
    N = 40
    some_neighs = geom.neighbors[N][0:3]  # pick 4 neighbors
    image[N] = 5.0              # set a single image pixel
    image[some_neighs] = 3.0    # make some boundaries that are neighbors
    image[10] = 3.0             # a boundary that is not a neighbor

    mask = tailcuts_clean(geom, image, pedvar)

    print((mask > 0).sum(), "clean pixels")
    print(geom.pix_id[mask])
