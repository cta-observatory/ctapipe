import logging

import numpy as np
from astropy import units as u

logger = logging.getLogger(__name__)

__all__ = ["convert_rect_image_1d_to_2d", "convert_rect_image_back_to_1d"]


def pos_to_index(pos, size):
    """
    Bin pixel positions on a grid with bin widths at least half the pixel size.
    This can be used to infer the rows and columns of square pixels.
    """
    rnd = np.round((pos / size).to_value(u.dimensionless_unscaled), 1)
    unique = np.sort(np.unique(rnd))
    mask = np.append(np.diff(unique) > 0.5, True)
    bins = np.append(unique[mask] - 0.5, unique[-1] + 0.5)
    return np.digitize(rnd, bins) - 1


def convert_rect_image_1d_to_2d(geom, image_flat):
    """
    Convert a 1-dimensional image to a 2-dimensional array
    so that normal image manipulation routines can be used.
    Depending on the camera geoemtrie

    Parameters:
    -----------
    geom: CameraGeometry object
        geometry object of hexagonal cameras
    image: ndarray
        1D array of the pmt signals

    Returns:
    --------
    (rows, cols): (ndarray, ndarray)
        Arrays holding the row and col of each pixel, needed to transform back
    image_2d: ndarray
        Square image
    """
    size = np.sqrt(geom.pix_area)
    col = pos_to_index(geom.pix_x, size)
    row = pos_to_index(geom.pix_y, size)
    image_square = np.full((row.max() + 1, col.max() + 1), np.nan)
    image_square[row, col] = image_flat

    return (row, col), np.flip(image_square, axis=0)


def convert_rect_image_back_to_1d(rows_cols, image_square):
    """
    Convert a 2-dimensional image back to a 1-dimensional array.

    Parameters:
    -----------
    rows_cols: (ndarray, ndarray)
        Row and column indices for each pixel
    image_2d: ndarray
        2d array of the PMT signals

    Returns:
    --------
    image_1d: ndarray
        The flattened camera image
    """
    image_flat = np.zeros_like(rows_cols[0], dtype=image_square.dtype)
    image_flat[:] = np.flip(image_square, axis=0)[rows_cols]
    return image_flat
