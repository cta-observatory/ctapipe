"""
Leakage calculation
"""

import numpy as np
from ..io.containers import LeakageContainer


__all__ = ["leakage"]


def leakage(geom, image, cleaning_mask):
    """
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
    """
    border1 = geom.get_border_pixel_mask(1)
    border2 = geom.get_border_pixel_mask(2)

    mask1 = border1 & cleaning_mask
    mask2 = border2 & cleaning_mask

    leakage_pixel1 = np.count_nonzero(mask1)
    leakage_pixel2 = np.count_nonzero(mask2)

    leakage_intensity1 = np.sum(image[mask1])
    leakage_intensity2 = np.sum(image[mask2])

    size = np.sum(image[cleaning_mask])

    return LeakageContainer(
        pixels_width_1=leakage_pixel1 / geom.n_pixels,
        pixels_width_2=leakage_pixel2 / geom.n_pixels,
        intensity_width_1=leakage_intensity1 / size,
        intensity_width_2=leakage_intensity2 / size,
    )
