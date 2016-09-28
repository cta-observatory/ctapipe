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


def ring_completeness(
        pixel_x,
        pixel_y,
        weights,
        center_x,
        center_y,
        radius,
        threshold=30,
        bins=30,
        ):
    angle = np.arctan2(pixel_y - center_y, pixel_x - center_x)
    hist, edges = np.histogram(angle, bins=bins, range=[-np.pi, np.pi], weights=weights)

    highest_bin = np.argmax(hist)
    if highest_bin < threshold:
        return 0

    bins_above_threshold = hist > threshold

    return np.sum(bins_above_threshold) / bins


