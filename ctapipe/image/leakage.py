"""
Leakage calculation
"""

import numpy as np
from ..io.containers import LeakageContainer


__all__ = ['leakage']


def leakage(geom, image, cleaning_mask):
    '''
    Calculating the leakage-values for a given image.
    Image must be cleaned for example with tailcuts_clean.
    Leakage describes how strong a shower is on the edge of a telescope.

    Parameters
    ----------
    geom: ctapipe.instrument.CameraGeometry
        Camera geometry information
    image: array
        pixel values
    cleaning_mask: array, dtype=bool
        The pixel that survived cleaning, e.g. tailcuts_clean

    Returns
    -------
    LeakageContainer
    '''
    border1 = geom.get_border_pixel_mask(1)
    border2 = geom.get_border_pixel_mask(2)

    mask1 = border1 & cleaning_mask
    mask2 = border2 & cleaning_mask

    leakage_pixel1 = np.sum(mask1)
    leakage_pixel2 = np.sum(mask2)

    leakage_intensity1 = np.sum(image[mask1])
    leakage_intensity2 = np.sum(image[mask2])

    size = np.sum(image[cleaning_mask])

    return LeakageContainer(
        leakage1_pixel=leakage_pixel1 / geom.n_pixels,
        leakage2_pixel=leakage_pixel2 / geom.n_pixels,
        leakage1_intensity=leakage_intensity1 / size,
        leakage2_intensity=leakage_intensity2 / size,
    )
