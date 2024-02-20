"""
Image timing-based shower image parametrization.
"""

import astropy.units as u
import numpy as np
from numba import njit

from ..containers import (
    CameraHillasParametersContainer,
    CameraTimingParametersContainer,
    HillasParametersContainer,
    TimingParametersContainer,
)
from ..fitting import lts_linear_regression
from ..utils.quantities import all_to_value
from .hillas import camera_to_shower_coordinates

__all__ = ["timing_parameters"]


@njit(cache=True)
def rmse(truth, prediction):
    """Root mean squared error"""
    return np.sqrt(np.mean((truth - prediction) ** 2))


def timing_parameters(geom, image, peak_time, hillas_parameters, cleaning_mask=None):
    """
    Function to extract timing parameters from a cleaned image.

    Parameters
    ----------
    geom: ctapipe.instrument.CameraGeometry
        Camera geometry
    image : array_like
        Pixel values
    peak_time : array_like
        Time of the pulse extracted from each pixels waveform
    hillas_parameters: ctapipe.containers.HillasParametersContainer
        Result of hillas_parameters
    cleaning_mask: optional, array, dtype=bool
        The pixels that survived cleaning, e.g. tailcuts_clean
        The non-masked pixels must verify signal > 0

    Returns
    -------
    timing_parameters: TimingParametersContainer
    """

    unit = geom.pix_x.unit

    # numba needs arguments to be the same type, so upcast to float64 if necessary
    peak_time = peak_time.astype(np.float64)

    if cleaning_mask is not None:
        image = image[cleaning_mask]
        geom = geom[cleaning_mask]
        peak_time = peak_time[cleaning_mask]

    if (image < 0).any():
        raise ValueError("The non-masked pixels must verify signal >= 0")

    h = hillas_parameters
    if isinstance(h, CameraHillasParametersContainer):
        unit = h.x.unit
        pix_x, pix_y, x, y, length, width = all_to_value(
            geom.pix_x, geom.pix_y, h.x, h.y, h.length, h.width, unit=unit
        )
    elif isinstance(h, HillasParametersContainer):
        unit = h.fov_lon.unit
        pix_x, pix_y, x, y, length, width = all_to_value(
            geom.pix_x, geom.pix_y, h.fov_lon, h.fov_lat, h.length, h.width, unit=unit
        )

    longi, _ = camera_to_shower_coordinates(
        pix_x, pix_y, x, y, hillas_parameters.psi.to_value(u.rad)
    )

    # re-fit using a robust-to-outlier algorithm
    beta, error = lts_linear_regression(x=longi, y=peak_time, samples=5)

    # error from lts_linear_regression is only for the used points,
    # recalculate for all points
    deviation = rmse(longi * beta[0] + beta[1], peak_time)

    if unit.is_equivalent(u.m):
        return CameraTimingParametersContainer(
            slope=beta[0] / unit, intercept=beta[1], deviation=deviation
        )
    return TimingParametersContainer(
        slope=beta[0] / unit, intercept=beta[1], deviation=deviation
    )
