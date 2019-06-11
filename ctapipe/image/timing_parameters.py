"""
Image timing-based shower image parametrization.
"""

import numpy as np
from numpy.polynomial.polynomial import polyfit, polyval
from ctapipe.io.containers import TimingParametersContainer
from .hillas import camera_to_shower_coordinates


__all__ = [
    'timing_parameters'
]


def timing_parameters(geom, image, pulse_time, hillas_parameters):
    """
    Function to extract timing parameters from a cleaned image

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

    Returns
    -------
    timing_parameters: TimingParametersContainer
    """

    unit = geom.pix_x.unit

    # select only the pixels in the cleaned image that are greater than zero.
    # we need to exclude possible pixels with zero signal after cleaning.
    greater_than_0 = image > 0
    pix_x = geom.pix_x[greater_than_0]
    pix_y = geom.pix_y[greater_than_0]
    image = image[greater_than_0]
    pulse_time = pulse_time[greater_than_0]

    longi, trans = camera_to_shower_coordinates(
        pix_x,
        pix_y,
        hillas_parameters.x,
        hillas_parameters.y,
        hillas_parameters.psi
    )
    intercept, slope = polyfit(
        longi.value, pulse_time, deg=1, w=np.sqrt(image)
    )
    predicted_time = polyval(longi.value, (intercept, slope))
    deviation = np.sqrt(
        np.sum((pulse_time - predicted_time)**2) / pulse_time.size
    )

    return TimingParametersContainer(
        slope=slope / unit,
        intercept=intercept,
        deviation=deviation,
    )
