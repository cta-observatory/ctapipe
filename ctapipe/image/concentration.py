import numpy as np
from numpy import nan

from .hillas import camera_to_shower_coordinates
from ..core import Container, Field


class ConcentrationContainer(Container):
    """
    Concentrations are ratios between light amount
    in certain areas of the image and the full image.
    """

    container_prefix = "concentration"
    cog = Field(
        nan, "Percentage of photo-electrons in the three pixels closest to the cog"
    )
    core = Field(nan, "Percentage of photo-electrons inside the hillas ellipse")
    pixel = Field(nan, "Percentage of photo-electrons in the brightest pixel")


def concentration(geom, image, hillas_parameters):
    """
    Calculate concentraion values.

    Concentrations are ratios of the amount of light in certain
    areas to the full intensity of the image.

    These features are usefull for g/h separation and energy estimation.
    """

    h = hillas_parameters

    delta_x = geom.pix_x - h.x
    delta_y = geom.pix_y - h.y

    # sort pixels by distance to cog
    cog_pixels = np.argsort(delta_x ** 2 + delta_y ** 2)
    conc_cog = np.sum(image[cog_pixels[:3]]) / h.intensity

    longi, trans = camera_to_shower_coordinates(geom.pix_x, geom.pix_y, h.x, h.y, h.psi)

    # get all pixels inside the hillas ellipse
    mask_core = (longi ** 2 / h.length ** 2) + (trans ** 2 / h.width ** 2) <= 1.0
    conc_core = image[mask_core].sum() / h.intensity

    concentration_pixel = image.max() / h.intensity

    return ConcentrationContainer(
        cog=conc_cog,
        core=conc_core,
        pixel=concentration_pixel,
    )
