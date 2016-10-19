import numpy as np
import logging

log = logging.getLogger(__name__)


def mean_squared_error(pixel_x, pixel_y, weights, radius, center_x, center_y):
    '''
    Calculate the weighted mean squared error for a circle

    Parameters
    ----------
    pixel_x: array-like
        x coordinates of the camera pixels
    pixel_y: array-like
        y coordinates of the camera pixels
    weights: array-like
        weights for the camera pixels, will usually be the pe charges
    radius: float
        radius of the ring
    center_x: float
        x coordinate of the ring center
    center_y: float
        y coordinate of the ring center
    '''
    r = np.sqrt((center_x - pixel_x)**2 + (center_y - pixel_y)**2)
    return np.average((r - radius)**2, weights=weights)


def photon_ratio_inside_ring(
        pixel_x, pixel_y, weights, radius, center_x, center_y, width
        ):
    '''
    Calculate the ratio of the photons inside a given ring with
    coordinates (center_x, center_y), radius and width.

    The ring is assumed to be in [radius - 0.5 * width, radius + 0.5 * width]

    Parameters
    ----------
    pixel_x: array-like
        x coordinates of the camera pixels
    pixel_y: array-like
        y coordinates of the camera pixels
    weights: array-like
        weights for the camera pixels, will usually be the pe charges
    radius: float
        radius of the ring
    center_x: float
        x coordinate of the ring center
    center_y: float
        y coordinate of the ring center
    width: float
        width of the ring
    '''

    total = np.sum(weights)

    pixel_r = np.sqrt((center_x - pixel_x)**2 + (center_y - pixel_y)**2)
    mask = np.logical_and(
        pixel_r >= radius - 0.5 * width,
        pixel_r <= radius + 0.5 * width
    )

    inside = np.sum(weights[mask])

    return inside / total


def ring_completeness(
        pixel_x,
        pixel_y,
        weights,
        radius,
        center_x,
        center_y,
        threshold=30,
        bins=30,
        ):
    '''
    Estimate how complete a ring is.
    Bin the light distribution along the the ring and apply a threshold to the
    bin content.

    Parameters
    ----------
    pixel_x: array-like
        x coordinates of the camera pixels
    pixel_y: array-like
        y coordinates of the camera pixels
    weights: array-like
        weights for the camera pixels, will usually be the pe charges
    radius: float
        radius of the ring
    center_x: float
        x coordinate of the ring center
    center_y: float
        y coordinate of the ring center
    threshold: float
        number of photons a bin must contain to be counted
    bins: int
        number of bins to use for the histogram

    Returns
    -------
    ring_completeness: float
        the ratio of bins above threshold

    Returns
    -------
    ring_completeness: float
        the ratio of bins above threshold
    '''

    angle = np.arctan2(pixel_y - center_y, pixel_x - center_x)

    hist, edges = np.histogram(angle, bins=bins, range=[-np.pi, np.pi], weights=weights)

    bins_above_threshold = hist > threshold

    return np.sum(bins_above_threshold) / bins
