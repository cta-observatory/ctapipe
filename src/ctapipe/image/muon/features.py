import astropy.units as u
import numpy as np

from ...utils.quantities import all_to_value

__all__ = [
    "mean_squared_error",
    "intensity_ratio_inside_ring",
    "ring_completeness",
    "ring_containment",
]


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
    r = np.sqrt((center_x - pixel_x) ** 2 + (center_y - pixel_y) ** 2)
    return np.average((r - radius) ** 2, weights=weights)


def intensity_ratio_inside_ring(
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

    pixel_r = np.sqrt((center_x - pixel_x) ** 2 + (center_y - pixel_y) ** 2)
    mask = np.logical_and(
        pixel_r >= radius - 0.5 * width, pixel_r <= radius + 0.5 * width
    )

    inside = weights[mask].sum()
    total = weights.sum()

    return inside / total


def ring_completeness(
    pixel_x, pixel_y, weights, radius, center_x, center_y, threshold=30, bins=30
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
    if hasattr(angle, "unit"):
        angle = angle.to_value(u.rad)

    hist, _ = np.histogram(angle, bins=bins, range=[-np.pi, np.pi], weights=weights)

    bins_above_threshold = hist > threshold

    return np.sum(bins_above_threshold) / bins


def ring_containment(radius, center_x, center_y, camera_radius):
    """
    Estimate angular containment of a ring inside the camera
    (camera center is (0,0))

    Improve: include the case of an arbitrary
    center for the camera

    See https://stackoverflow.com/questions/3349125/circle-circle-intersection-points

    Parameters
    ----------
    radius: float or quantity
        radius of the muon ring
    center_x: float or quantity
        x coordinate of the center of the muon ring
    center_y: float or quantity
        y coordinate of the center of the muon ring
    camera_radius: float or quantity
        radius of the camera

    Returns
    -------
    ringcontainment: float
        the ratio of ring inside the camera
    """
    if hasattr(radius, "unit"):
        radius, center_x, center_y, camera_radius = all_to_value(
            radius, center_x, center_y, camera_radius, unit=radius.unit
        )
    d = np.sqrt(center_x**2 + center_y**2)

    # one circle fully contained in the other
    if d <= np.abs(camera_radius - radius):
        return 1.0

    # no intersection
    if d > (radius + camera_radius):
        return 0.0

    a = (radius**2 - camera_radius**2 + d**2) / (2 * d)
    return np.arccos(a / radius) / np.pi
