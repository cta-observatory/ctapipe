"""
Image timing-based shower image parametrization.
"""

import numpy as np
from numpy.polynomial.polynomial import polyval
from ctapipe.io.containers import TimingParametersContainer
from .hillas import camera_to_shower_coordinates


__all__ = [
    'timing_parameters'
]


def timing_parameters(geom, image, pulse_time, hillas_parameters, cleaning_mask=None):
    """
    Function to extract timing parameters from a cleaned image.

    Parameters
    ----------
    geom: ctapipe.instrument.CameraGeometry
        Camera geometry
    image : array_like
        Pixel values
    pulse_time : array_like
        Time of the pulse extracted from each pixels waveform
    hillas_parameters: ctapipe.io.containers.HillasParametersContainer
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
        pulse_time = pulse_time[cleaning_mask]

    if (image < 0).any():
        raise ValueError("The non-masked pixels must verify signal >= 0")

    pix_x = geom.pix_x
    pix_y = geom.pix_y

    longi, trans = camera_to_shower_coordinates(
        pix_x,
        pix_y,
        hillas_parameters.x,
        hillas_parameters.y,
        hillas_parameters.psi
    )
    (slope, intercept), cov = np.polyfit(
        longi.value, pulse_time, deg=1, w=np.sqrt(image), cov=True,
    )
    slope_err, intercept_err = np.sqrt(np.diag(cov))
    predicted_time = polyval(longi.value, (intercept, slope))
    deviation = np.sqrt(
        np.sum((pulse_time - predicted_time)**2) / pulse_time.size
    )

    return TimingParametersContainer(
        slope=slope / unit,
        intercept=intercept,
        deviation=deviation,
        slope_err=slope_err,
        intercept_err=intercept_err,
    )
