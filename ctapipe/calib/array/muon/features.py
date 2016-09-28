import numpy as np


def mean_squared_error(pixel_x, pixel_y, weights, center_x, center_y, radius):
    ''' calculate the weighted mean squared error for a circle '''
    r = np.sqrt((center_x - pixel_x)**2 + (center_y - pixel_y)**2)
    return np.average((r - radius)**2, weights=weights)


def photon_ratio_inside_ring(
        pixel_x, pixel_y, weights, center_x, center_y, radius, width
        ):
    '''
    Calculate the ratio of the photons inside a given ring with
    coordinates (center_x, center_y), radius and width.
    '''

    total = np.sum(weights)

    pixel_r = np.sqrt((center_x - pixel_x)**2 + (center_y - pixel_y)**2)
    mask = np.logical_and(
        pixel_r >= radius - width,
        pixel_r <= radius + width
    )

    inside = np.sum(weights[mask])

    return inside / total



