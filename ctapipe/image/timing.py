"""
Image timing-based shower image parametrization.
"""

import numpy as np
import astropy.units as u
from numba import njit

from ..containers import TimingParametersContainer
from .hillas import camera_to_shower_coordinates
from ..utils.quantities import all_to_value


__all__ = ["timing_parameters"]


@njit(cache=True)
def linear_regression(x, y):
    """
    njit version of a least squares linear regression

    Parameters
    ----------

    x: np.ndarray
        x values
    y: np.ndarray
        y values

    Returns
    -------
    slope: float
        slope of the linear regression result
    intercept: float
        intercept of the linear regression result
    cov: 2x2 ndarray
        covariance matrix of the slope and intercept
    """
    y = y.reshape((-1, 1))
    X = np.empty((len(x), 2))
    X[:, 0] = x
    X[:, 1] = 1

    cov = np.linalg.inv(X.T @ X)
    params = cov @ X.T @ y
    return params[0], params[1], cov


@njit(cache=True)
def sigma_clipping_linreg(x, y, kappa=3, n_iter=3):
    """
    Linear regression with sigma clipping.
    Iteratively perform a linear regression, excluding points
    further away than ``kappa`` times the current root mean squared error.

    Parameters
    ----------

    x: np.ndarray
        x values
    y: np.ndarray
        y values
    kappa: float
        maximum distance from fit in terms of rmse
    n_iter: int
        How many iterations of sigma clipping to perform

    Returns
    -------
    slope: float
        slope of the linear regression result
    intercept: float
        intercept of the linear regression result
    cov: 2x2 ndarray
        covariance matrix of the slope and intercept
    """

    a, b, cov = linear_regression(x, y)

    for i in range(n_iter):
        delta = y - (a * x + b)
        sigma = np.std(delta)

        mask = delta < (kappa * sigma)
        a, b, cov = linear_regression(x[mask], y[mask])

    return a, b, cov


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
    cleaning_mask: optionnal, array, dtype=bool
        The pixels that survived cleaning, e.g. tailcuts_clean
        The non-masked pixels must verify signal > 0

    Returns
    -------
    timing_parameters: TimingParametersContainer
    """

    unit = geom.pix_x.unit

    if cleaning_mask is not None:
        image = image[cleaning_mask]
        geom = geom[cleaning_mask]
        peak_time = peak_time[cleaning_mask]

    if (image < 0).any():
        raise ValueError("The non-masked pixels must verify signal >= 0")

    h = hillas_parameters
    pix_x, pix_y, x, y, length, width = all_to_value(
        geom.pix_x, geom.pix_y, h.x, h.y, h.length, h.width, unit=unit
    )

    longi, _ = camera_to_shower_coordinates(
        pix_x, pix_y, x, y, hillas_parameters.psi.to_value(u.rad)
    )

    slope, intercept, cov = sigma_clipping_linreg(
        x=longi, y=peak_time.astype("float64")
    )
    slope_err, intercept_err = np.sqrt(np.diag(cov))

    predicted_time = slope * longi + intercept
    rsme = np.sqrt(np.mean((peak_time - predicted_time) ** 2))

    return TimingParametersContainer(
        slope=slope / unit,
        intercept=intercept,
        deviation=rsme,
        slope_err=slope_err / unit,
        intercept_err=intercept_err,
    )
