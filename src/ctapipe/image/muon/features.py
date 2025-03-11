from typing import Sequence

import astropy.units as u
import numpy as np
from astropy.units import Quantity

from ...containers import MuonRingContainer

__all__ = [
    "mean_squared_error",
    "intensity_ratio_inside_ring",
    "ring_completeness",
    "ring_containment",
    "ring_intensity_parameters",
    "radial_light_distribution",
]


def mean_squared_error(
    pixel_fov_lon: Quantity,
    pixel_fov_lat: Quantity,
    weights: Quantity | np.ndarray | Sequence[float],
    ring: MuonRingContainer,
) -> float:
    """
    Calculate the weighted mean squared error for a circle.

    Parameters
    ----------
    pixel_fov_lon : Quantity
        Longitudes (x-coordinates) of the camera pixels in the TelescopeFrame.
    pixel_fov_lat : Quantity
        Latitudes (y-coordinates) of the camera pixels in the TelescopeFrame.
    weights : Quantity | np.ndarray | Sequence[float]
        Weights for the camera pixels, usually the photoelectron charges.
    ring : MuonRingContainer
        Container with the fitted ring parameters, including center coordinates and radius.

    Returns
    -------
    float
        The weighted mean squared error of the pixels around the fitted ring.

    Notes
    -----
    This function calculates the weighted mean squared error of the pixels around
    the fitted ring by determining the radial distance of each pixel from the ring
    center and comparing it to the ring radius. The mean squared error is weighted
    by the pixel intensities.
    """
    r = np.hypot(
        ring.center_fov_lon - pixel_fov_lon, ring.center_fov_lat - pixel_fov_lat
    )
    return np.average((r - ring.radius) ** 2, weights=weights)


def intensity_ratio_inside_ring(
    pixel_fov_lon: Quantity,
    pixel_fov_lat: Quantity,
    weights: Quantity | np.ndarray | Sequence[float],
    ring: MuonRingContainer,
    width: Quantity,
) -> float:
    """
    Calculate the ratio of the photons inside a given ring with
    coordinates (center_fov_lon, center_fov_lat), radius and width.

    The ring is assumed to be in [radius - 0.5 * width, radius + 0.5 * width]

    Parameters
    ----------
    pixel_fov_lon : Quantity
        Longitudes (x-coordinates) of the camera pixels in the TelescopeFrame.
    pixel_fov_lat : Quantity
        Latitudes (y-coordinates) of the camera pixels in the TelescopeFrame.
    weights : Quantity | np.ndarray | Sequence[float]
        Weights for the camera pixels, usually the photoelectron charges.
    ring : MuonRingContainer
        Container with the fitted ring parameters, including center coordinates and radius.
    width : Quantity
        Width of the ring.

    Returns
    -------
    float
        The ratio of the photons inside the ring to the total photons.

    Notes
    -----
    This function calculates the ratio of the photons inside a given ring by
    determining the pixels that fall within the specified ring width and summing
    their weights. The ratio is the sum of the weights inside the ring divided by
    the total sum of the weights.
    """

    pixel_r = np.hypot(
        ring.center_fov_lon - pixel_fov_lon, ring.center_fov_lat - pixel_fov_lat
    )
    mask = np.logical_and(
        pixel_r >= ring.radius - 0.5 * width, pixel_r <= ring.radius + 0.5 * width
    )

    inside = weights[mask].sum()
    total = weights.sum()

    return inside / total


def ring_completeness(
    pixel_fov_lon: Quantity,
    pixel_fov_lat: Quantity,
    weights: Quantity | np.ndarray | Sequence[float],
    ring: MuonRingContainer,
    threshold: float = 30,
    bins: int = 30,
) -> float:
    """
    Estimate how complete a muon ring is by binning the light distribution along the ring
    and applying a threshold to the bin content.

    Parameters
    ----------
    pixel_fov_lon : Quantity
        Longitudes (x-coordinates) of the camera pixels in the TelescopeFrame.
    pixel_fov_lat : Quantity
        Latitudes (y-coordinates) of the camera pixels in the TelescopeFrame.
    weights : array-like
        Weights for the camera pixels, usually the photoelectron charges.
    ring : MuonRingContainer
        Container with the fitted ring parameters, including center coordinates and radius.
    threshold : float, optional
        Number of photoelectrons a bin must contain to be counted. Default is 30.
    bins : int, optional
        Number of bins to use for the histogram. Default is 30.

    Returns
    -------
    float
        The ratio of bins above the threshold, representing the completeness of the ring.

    Notes
    -----
    This function calculates the completeness of the muon ring by dividing the ring into
    segments and counting the number of segments that have a light intensity above a given
    threshold. The completeness is the ratio of the number of segments above the threshold
    to the total number of segments.
    """

    if hasattr(weights, "unit"):
        weights = weights.to_value(u.dimensionless_unscaled)
    angle = np.arctan2(
        (pixel_fov_lat - ring.center_fov_lat).to_value(u.rad),
        (pixel_fov_lon - ring.center_fov_lon).to_value(u.rad),
    )

    hist, _ = np.histogram(
        angle,
        bins=bins,
        range=[-np.pi, np.pi],
        weights=weights,
    )

    bins_above_threshold = hist > threshold

    return np.sum(bins_above_threshold) / bins


def ring_containment(ring: MuonRingContainer, camera_radius: Quantity) -> float:
    """
    Estimate the angular containment of a muon ring inside the camera's field of view.

    This function calculates the fraction of the muon ring that is contained within
    the camera's field of view. It uses geometric properties to determine the intersection
    of the ring with the circular boundary of the camera.

    Parameters
    ----------
    ring : MuonRingContainer
        Container with the fitted ring parameters, including center coordinates and radius.
    camera_radius : `astropy.units.Quantity`
        Radius of the camera's field of view (in degrees).

    Returns
    -------
    float
        The fraction of the ring that is inside the camera's field of view, ranging from 0.0 to 1.0.

    Notes
    -----
    The calculation is based on the geometric intersection of two circles:
    the muon ring and the camera's field of view. The method handles cases where:
    - The ring is fully contained within the camera.
    - The ring is partially contained within the camera.
    - The ring is completely outside the camera.

    References
    ----------
    See https://stackoverflow.com/questions/3349125/circle-circle-intersection-points
    for the geometric approach to circle-circle intersection.
    """
    # one circle fully contained in the other
    if ring.center_distance <= np.abs(camera_radius - ring.radius):
        return 1.0

    # no intersection
    if ring.center_distance > (ring.radius + camera_radius):
        return 0.0

    a = (ring.radius**2 - camera_radius**2 + ring.center_distance**2) / (
        2 * ring.center_distance
    )
    return np.arccos((a / ring.radius).to_value(u.dimensionless_unscaled)) / np.pi


def ring_intensity_parameters(
    ring: MuonRingContainer,
    pixel_fov_lon: Quantity,
    pixel_fov_lat: Quantity,
    ring_integration_width: float,
    outer_ring_width: float,
    image: np.ndarray,
    image_mask: np.ndarray,
) -> tuple[float, float, int, float]:
    """
    Calculate the parameters related to the size of the ring image.

    Parameters
    ----------
    ring : MuonRingContainer
        Container with the fitted ring parameters, including center coordinates and radius.
    pixel_fov_lon : Quantity
        Longitudes (x-coordinates) of the camera pixels in the TelescopeFrame.
    pixel_fov_lat : Quantity
        Latitudes (y-coordinates) of the camera pixels in the TelescopeFrame.
    ring_integration_width : float
        Width of the ring in fractions of ring radius.
    outer_ring_width : float
        Width of the outer ring in fractions of ring radius.
    image : np.ndarray
        Amplitude of image pixels.
    image_mask : np.ndarray
        Mask of the camera pixels after cleaning.

    Returns
    -------
    ring_intensity : float
        Sum of the p.e. inside the integration area of the ring.
    intensity_outside_ring : float
        Sum of the p.e. outside the ring integration area that passed the cleaning.
    n_pixels_in_ring : int
        Number of pixels inside the ring integration area that passed the cleaning.
    mean_intensity_outside_ring : float
        Mean intensity of the pixels outside the integration area of the ring,
        and restricted by the outer ring width, i.e. in the strip between
        ring integration width and outer ring width.
    """

    dist = np.hypot(
        pixel_fov_lon - ring.center_fov_lon, pixel_fov_lat - ring.center_fov_lat
    )
    dist_mask = np.abs(dist - ring.radius) < (ring.radius * ring_integration_width)
    pix_ring = image * dist_mask
    pix_outside_ring = image * ~dist_mask

    dist_mask_2 = np.logical_and(
        ~dist_mask,
        np.abs(dist - ring.radius)
        < ring.radius * (ring_integration_width + outer_ring_width),
    )
    pix_ring_2 = image[dist_mask_2]

    ring_intensity = np.sum(pix_ring)
    intensity_outside_ring = np.sum(pix_outside_ring * image_mask)
    n_pixels_in_ring = np.sum(dist_mask & image_mask)
    mean_intensity_outside_ring = np.sum(pix_ring_2) / len(pix_ring_2)

    return (
        ring_intensity,
        intensity_outside_ring,
        n_pixels_in_ring,
        mean_intensity_outside_ring,
    )


def radial_light_distribution(
    center_fov_lon, center_fov_lat, pixel_fov_lon, pixel_fov_lat, image
):
    """
    Calculate the radial distribution of the muon ring.

    Parameters
    ----------
    center_fov_lon : float
        Longitude of the ring center in the TelescopeFrame (in degrees).
    center_fov_lat : float
        Latitude of the ring center in the TelescopeFrame (in degrees).
    pixel_fov_lon : array-like
        Longitudes (x-coordinates) of the camera pixels in the TelescopeFrame (in degrees).
    pixel_fov_lat : array-like
        Latitudes (y-coordinates) of the camera pixels in the TelescopeFrame (in degrees).
    image : array-like
        Amplitudes of image pixels.

    Returns
    -------
    radial_std_dev : `astropy.units.Quantity`
        Standard deviation of the light distribution in degrees.
        Spread of pixel intensities around the mean radial distance from the ring center.
    skewness : `astropy.units.Quantity`
        Skewness of the radial light distribution (dimensionless).
        Measures the asymmetry of the distribution around the mean radius.
    excess_kurtosis : `astropy.units.Quantity`
        Excess kurtosis of the radial light distribution (dimensionless).
        Indicates the "tailedness" of the distribution compared to a normal distribution.

    Notes
    -----
    This function calculates the statistical properties of the radial distribution
    of light in an image with respect to the reconstructed muon ring center. It computes
    the standard deviation, skewness, and excess kurtosis of the radial distances of the pixels
    from the center of the ring, weighted by the pixel intensities.
    """

    if np.sum(image) == 0:
        return (
            np.nan * u.deg,
            np.nan * u.dimensionless_unscaled,
            np.nan * u.dimensionless_unscaled,
        )

    pixel_r = np.hypot(pixel_fov_lon - center_fov_lon, pixel_fov_lat - center_fov_lat)

    mean = np.average(pixel_r, weights=image)
    delta_r = pixel_r - mean
    radial_std_dev = np.sqrt(np.average(delta_r**2, weights=image))
    skewness = (
        np.average(delta_r**3, weights=image)
        / radial_std_dev**3
        * u.dimensionless_unscaled
    )
    excess_kurtosis = (
        np.average(delta_r**4, weights=image) / radial_std_dev**4 - 3.0
    ) * u.dimensionless_unscaled

    return (
        radial_std_dev,
        skewness,
        excess_kurtosis,
    )
