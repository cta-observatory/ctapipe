"""
Image timing-based shower image parametrization.
"""

import numpy as np
from ctapipe.io.containers import TimingParametersContainer

__all__ = [
    'timing_parameters'
]


def timing_parameters(geom, image, peakpos, hillas_parameters):
    """
    Function to extract timing parameters from a cleaned image

    Parameters
    ----------
    geom: ctapipe.instrument.CameraGeometry
        Camera geometry
    image : array_like
        Pixel values
    peakpos : array_like
        Pixel peak positions array
    hillas_parameters: ctapipe.io.containers.HillasParametersContainer
        Result of hillas_parameters

    Returns
    -------
    timing_parameters: TimingParametersContainer
    """

    unit = geom.pix_x.unit
    pix_x = geom.pix_x.value
    pix_y = geom.pix_y.value

    # select only the pixels in the cleaned image that are greater than zero.
    # This is to allow to use a dilated mask (which might be better):
    # we need to exclude possible pixels with zero signal after cleaning.

    mask = np.ma.masked_where(image > 0, image).mask
    pix_x = pix_x[mask]
    pix_y = pix_y[mask]
    image = image[mask]
    peakpos = peakpos[mask]

    assert pix_x.shape == image.shape
    assert pix_y.shape == image.shape
    assert peakpos.shape == image.shape

    longi, trans = geom.get_shower_coordinates(
        hillas_parameters.x,
        hillas_parameters.y,
        hillas_parameters.psi
    )
    slope, intercept = np.polyfit(longi, peakpos, deg=1, w=np.sqrt(image))

    return TimingParametersContainer(
        slope=slope / unit,
        intercept=intercept,
    )
