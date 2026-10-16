import astropy.units as u
import numpy as np

from ..containers import ConcentrationContainer
from ..instrument import CameraGeometry
from ..utils.quantities import all_to_value
from .hillas import camera_to_shower_coordinates

__all__ = ["concentration_parameters"]


def concentration_parameters(geom: CameraGeometry, image, hillas_parameters):
    """
    Calculate concentration values.

    Concentrations are ratios of the amount of light in certain
    areas to the full intensity of the image.

    These features are useful for g/h separation and energy estimation.

    Parameters
    ----------
    geom : ctapipe.instrument.CameraGeometry
        Camera geometry transformed to
        `~ctapipe.coordinates.TelescopeFrame`. The pixel coordinates must have
        angular units and use the same frame as ``hillas_parameters``.
    image : array_like
        Charge in each pixel after image cleaning.
    hillas_parameters : ctapipe.containers.HillasParametersContainer
        Result of :func:`~ctapipe.image.hillas_parameters` for ``geom``, with
        angular coordinates in the same telescope frame.

    Returns
    -------
    ctapipe.containers.ConcentrationContainer
        Container holding the concentration ratios.

    Raises
    ------
    astropy.units.UnitConversionError
        If the geometry coordinates and Hillas parameters do not have compatible
        angular units.
    """

    h = hillas_parameters
    unit = h.fov_lon.unit
    pix_x, pix_y, x, y, length, width, pixel_width = all_to_value(
        geom.pix_x,
        geom.pix_y,
        h.fov_lon,
        h.fov_lat,
        h.length,
        h.width,
        geom.pixel_width,
        unit=unit,
    )

    delta_x = pix_x - x
    delta_y = pix_y - y

    # take pixels within one pixel diameter from the cog
    mask_cog = (delta_x**2 + delta_y**2) < pixel_width**2
    conc_cog = np.sum(image[mask_cog]) / h.intensity

    if hillas_parameters.width.value != 0:
        # get all pixels inside the hillas ellipse
        longi, trans = camera_to_shower_coordinates(
            pix_x, pix_y, x, y, h.psi.to_value(u.rad)
        )
        mask_core = (longi**2 / length**2) + (trans**2 / width**2) <= 1.0
        conc_core = image[mask_core].sum() / h.intensity
    else:
        conc_core = 0.0

    concentration_pixel = image.max() / h.intensity

    return ConcentrationContainer(
        cog=conc_cog, core=conc_core, pixel=concentration_pixel
    )
