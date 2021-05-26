import logging

import numpy as np
from astropy import units as u


logger = logging.getLogger(__name__)

__all__ = ["convert_rect_image_1d_to_2d", "convert_rect_image_back_to_1d"]


def pos_to_index(pos, size):
    rnd = np.round((pos / size).to_value(u.dimensionless_unscaled), 1)
    unique = np.sort(np.unique(rnd))
    mask = np.append(np.diff(unique) > 0.5, True)
    bins = np.append(unique[mask] - 0.5, unique[-1] + 0.5)
    return np.digitize(rnd, bins) - 1


def convert_rect_image_1d_to_2d(image, geom):
    size = np.sqrt(geom.pix_area)
    col = pos_to_index(geom.pix_x, size)
    row = pos_to_index(geom.pix_y, size)
    img_sq = np.full((row.max() + 1, col.max() + 1), np.nan)
    img_sq[row, col] = image

    return np.flip(img_sq, axis=0), row, col


def convert_rect_image_back_to_1d(image, row, col):
    image_flat = np.zeros_like(row)
    image_flat[:] = np.flip(image, axis=0)[row, col]
    return image_flat
