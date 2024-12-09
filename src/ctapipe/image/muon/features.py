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


def ring_size_parameters(
    radius, center_x, center_y, pixel_x, pixel_y, ring_integration_width, outer_ring_width, image, image_mask
):
    """
    Calculate the parameters related to the size of the ring image.

    Parameters
    ----------
    radius: float
        radius of the ring
    center_x: float
        x coordinate of the ring center
    center_y: float
        y coordinate of the ring center
    pixel_x: array-like
        x coordinates of the camera pixels
    pixel_y: array-like
        y coordinates of the camera pixels
    ring_integration_width: float
        Width of the ring in fractions of ring radius
    outer_ring_width: float
        Width of the outer ring in fractions of ring radius
    image: array-like
        Amplitude of image pixels
    image_mask: array-like
        mask of the camera pixels after cleaning

    Returns
    -------
    ring_size: float
        Sum of the p.e. inside the integration area of the ring
    size_outside: float
        Sum of the photons outside the ring integration area that passed the cleaning
    num_pixels_in_ring: int
        Number of pixels inside the ring integration area that passed the cleaning
    mean_pixel_outside_ring: float
        Mean intensity of the pixels outside the ring, but still close to it
    """

    dist = np.sqrt((pixel_x - center_x) ** 2 + (pixel_y - center_y) ** 2)
    dist_mask = np.abs(dist - radius) < (radius * ring_integration_width)
    pix_ring = image * dist_mask
    pix_outside_ring = image * ~dist_mask
    
    dist_mask_2 = np.logical_and(~dist_mask,
                                    np.abs(dist - radius) <
                                    radius *
                                    (ring_integration_width + outer_ring_width)
                                )
    pix_ring_2 = image[dist_mask_2]
    
    ring_size = np.sum(pix_ring)
    size_outside = np.sum(pix_outside_ring * image_mask)
    num_pixels_in_ring = np.sum(dist_mask & image_mask)
    mean_pixel_outside_ring = (np.sum(pix_ring_2) / len(pix_ring_2))
    
    return ring_size, size_outside, num_pixels_in_ring, mean_pixel_outside_ring


def radial_light_distribution(center_x, center_y, pixel_x, pixel_y, image):
    """
    Calculate the radial distribution of the muon ring.

    Parameters
    ----------
    center_x : float
        x coordinate of the ring center.
    center_y : float
        y coordinate of the ring center.
    pixel_x : array-like
        x coordinates of the camera pixels.
    pixel_y : array-like
        y coordinates of the camera pixels.
        Amplitude of image pixels.

    Returns
    -------
    standard_dev : float
        Standard deviation of the light distribution along the ring radius.
    skewness : float
        Skewness of the light distribution along the ring radius.
    excess_kurtosis : float
        Excess kurtosis of the light distribution along the ring radius.
    """


    if np.sum(image) == 0:
        return np.nan * u.deg, np.nan, np.nan
    
    x0 = center_x.to_value(u.deg)
    y0 = center_y.to_value(u.deg)
    pix_x = pixel_x.to_value(u.deg)
    pix_y = pixel_y.to_value(u.deg)
    pixel_r = np.sqrt((pix_x - x0) ** 2 + (pix_y - y0) ** 2)

    mean = np.average(pixel_r, weights=image)
    delta_r = pixel_r - mean
    standard_dev = np.sqrt(np.average(delta_r ** 2, weights=image))
    skewness = np.average(delta_r ** 3, weights=image) / standard_dev ** 3
    excess_kurtosis = np.average(delta_r ** 4, weights=image) / standard_dev ** 4 - 3.

    return standard_dev * u.deg, skewness, excess_kurtosis
