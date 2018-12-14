import numpy as np
import logging
import math as mt

log = logging.getLogger(__name__)


def mean_squared_error(pixel_x, pixel_y, weights, radius, center_x, center_y):
    """
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
    """
    r = np.sqrt((center_x - pixel_x)**2 + (center_y - pixel_y)**2)
    return np.average((r - radius)**2, weights=weights)


def photon_ratio_inside_ring(
        pixel_x, pixel_y, weights, radius, center_x, center_y, width
        ):
    """
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
    """

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
    """
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
    """

    angle = np.arctan2(pixel_y - center_y, pixel_x - center_x)

    hist, edges = np.histogram(angle, bins=bins, range=[-np.pi, np.pi], weights=weights)

    bins_above_threshold = hist > threshold

    return np.sum(bins_above_threshold) / bins


def ring_containment(
        ring_radius,
        cam_rad,
        cring_x,
        cring_y,
        ):

    """
    Estimate angular containment of a ring inside the camera
    (camera center is (0,0))
    Improve: include the case of an arbitrary
    center for the camera

    Parameters
    ----------
    ring_radius: float
        radius of the muon ring
    cam_rad: float
        radius of the camera
    cring_x: float
        x coordinate of the center of the muon ring
    cring_y: float
        y coordinate of the center of the muon ring

    Returns
    ------
    ringcontainment: float
        the ratio of ring inside the camera
    """
    angle_ring = np.linspace(0, 2 * mt.pi, 360)
    ring_x = cring_x + ring_radius * np.cos(angle_ring)
    ring_y = cring_y + ring_radius * np.sin(angle_ring)
    d = np.sqrt(np.power(ring_x, 2) + np.power(ring_y, 2))

    ringcontainment = len(d[d < cam_rad]) / len(d)

    return ringcontainment


def npix_above_threshold(pix, thr):
    """
    Calculate number of pixels above a given threshold

    Parameters
    ----------
    pix: array-like
        array with pixel content, usually pe
    thr: float
        threshold for the pixels to be counted

    Returns
    ------
    npix_above_threshold: float
        Number of pixels above threshold
    """

    return (pix > thr).sum()


def npix_composing_ring(pix):
    """
    Calculate number of pixels composing a ring

    Parameters
    ----------
    pix: array-like
        array with pixel content, usually pe

    Returns
    ------
    npix_composing ring: float
        Number of pixels composing a ring
    """

    return np.count_nonzero(pix)
