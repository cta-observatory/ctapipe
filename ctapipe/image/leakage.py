"""
Leakage calculation
"""

import numpy as np
from ..io.containers import LeakageContainer


__all__ = ['leakage']


def leakage(geom, image):
    """Calculating the leakage-values for a given image.
    Image must be cleaned for example with tailcuts_clean.
    Leakage describes how strong a shower is on the edge of a telescope.

    Parameters
    ----------
    geom: ctapipe.instrument.CameraGeometry
        Camera geometry information
    image: array
        pixel values

    Returns
    -------
    leakage_pixel1:   float
        Number of shower-pixel on the border divided by all shower-pixel
    leakage_pixel2:   float
        Number of shower-pixel in the second row of the border
        divided by all shower-pixel
    leakage_intensity1:   float
        Number of photo-electrons in the border-pixel divided by all photo-electrons
    leakage_intensity2:   float
        Number of photo-electrons in the second row of the border-pixel
        divided by all photo-electrons

    """

    max_value = max(np.sum(geom.neighbor_matrix, axis=0))
    leakage_pixel1 = 0
    leakage_pixel2 = 0
    leakage_intensity1 = 0
    leakage_intensity2 = 0
    size = 0
    pixel_count = 0
    nonzero_index = np.nonzero(image)[0]
    for i in nonzero_index:
        size += image[i]
        pixel_count += 1
        if np.sum(geom.neighbor_matrix[i]) != max_value:
            leakage_pixel1 += 1
            leakage_intensity1 += image[i]
        else:
            nonzero_index2 = np.nonzero(geom.neighbor_matrix[i])[0]
            for j in nonzero_index2:
                if np.sum(geom.neighbor_matrix[j]) != max_value:
                    leakage_pixel2 += 1
                    leakage_intensity2 += image[i]
                    break
    return LeakageContainer(
        leakage_pixel1=leakage_pixel1 / pixel_count,
        leakage_pixel2=leakage_pixel2 / pixel_count,
        leakage_intensity1=leakage_intensity1 / size,
        leakage_intensity2=leakage_intensity2 / size,
    )
