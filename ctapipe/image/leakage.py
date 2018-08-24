"""
Leakage calculation
"""

import numpy as np
__all__ = ['leakage']


def leakage(geom, image):
    r"""Calculating the leakage-values for a given image.
    Image must be cleaned for example with tailcuts_clean.

    Parameters
    ----------
        geom: `ctapipe.instrument.CameraGeometry`
            Camera geometry information
        image: array
            pixel values

    Returns
    -------
        leakage_border_pixel1:   float
            Number of shower-pixel on the border divided by all shower-pixel
        leakage_border_pixel2:   float
            Number of shower-pixel in the second row of the border
             divided by all shower-pixel
        leakage_border_photon1:   float
            Number of photons in the border-pixel divided by all photons
        leakage_border_photon2:   float
            Number of photons in the second row of the border-pixel
             divided by all photons

    """

    max_value = max(np.sum(geom.neighbor_matrix, axis=0))
    leakage_border_pixel1 = 0
    leakage_border_pixel2 = 0
    leakage_border_photon1 = 0
    leakage_border_photon2 = 0
    size = 0
    pixel_count = 0
    nonzero_index = np.nonzero(image)[0]
    for i in nonzero_index:
        size += image[i]
        pixel_count += 1
        if np.sum(geom.neighbor_matrix[i]) != max_value:
            leakage_border_pixel1 += 1
            leakage_border_photon1 += image[i]
        for j in range(len(geom.neighbor_matrix[i])):
            if geom.neighbor_matrix[i][j] == 0:
                continue
            if np.sum(geom.neighbor_matrix[j]) != max_value:
                leakage_border_pixel2 += 1
                leakage_border_photon2 += image[i]
                break
    return leakage_border_pixel1 / pixel_count, \
           leakage_border_pixel2 / pixel_count, \
           leakage_border_photon1 / size, \
           leakage_border_photon2 / size
