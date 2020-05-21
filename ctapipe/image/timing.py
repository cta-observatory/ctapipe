"""
Image timing-based shower image parametrization.
"""

import numpy as np
import astropy.units as u
from numpy.polynomial.polynomial import polyval
from ..containers import TimingParametersContainer
from .hillas import camera_to_shower_coordinates
from ..utils.quantities import all_to_value

from scipy.stats import siegelslopes


__all__ = ["timing_parameters"]


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

    # use polyfit just to get the covariance matrix and errors
    (_s, _i), cov = np.polyfit(longi, peak_time, deg=1, w=np.sqrt(image), cov=True)
    slope_err, intercept_err = np.sqrt(np.diag(cov))

    # re-fit using a robust-to-outlier algorithm
    slope, intercept = siegelslopes(x=longi, y=peak_time)
    predicted_time = polyval(longi, (intercept, slope))
    deviation = np.sqrt(np.sum((peak_time - predicted_time) ** 2) / peak_time.size)

    return TimingParametersContainer(
        slope=slope / unit,
        intercept=intercept,
        deviation=deviation,
        slope_err=slope_err / unit,
        intercept_err=intercept_err,
    )
