import numpy as np
import astropy.units as u

from ..containers import ConcentrationContainer
from .hillas import camera_to_shower_coordinates
from ..utils.quantities import all_to_value

__all__ = ["concentration_parameters"]


def concentration_parameters(geom, image, hillas_parameters):
    """
    Calculate concentraion values.

    Concentrations are ratios of the amount of light in certain
    areas to the full intensity of the image.

    These features are usefull for g/h separation and energy estimation.
    """

    h = hillas_parameters
    unit = h.x.unit

    pix_x, pix_y, x, y, length, width = all_to_value(
        geom.pix_x, geom.pix_y, h.x, h.y, h.length, h.width, unit=unit
    )

    delta_x = pix_x - x
    delta_y = pix_y - y

    # sort pixels by distance to cog
    cog_pixels = np.argsort(delta_x ** 2 + delta_y ** 2)
    conc_cog = np.sum(image[cog_pixels[:3]]) / h.intensity

    if hillas_parameters.width.value != 0:
        # get all pixels inside the hillas ellipse
        longi, trans = camera_to_shower_coordinates(
            pix_x, pix_y, x, y, h.psi.to_value(u.rad)
        )
        mask_core = (longi ** 2 / length ** 2) + (trans ** 2 / width ** 2) <= 1.0
        conc_core = image[mask_core].sum() / h.intensity
    else:
        conc_core = 0.0

    concentration_pixel = image.max() / h.intensity

    return ConcentrationContainer(
        cog=conc_cog, core=conc_core, pixel=concentration_pixel
    )
