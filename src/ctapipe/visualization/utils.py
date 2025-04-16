import astropy.units as u
import numpy as np
from astropy.coordinates import Angle

from ..containers import HillasParametersContainer


def build_hillas_overlay(hillas, unit, with_label=True, n_sigma=1):
    """
    Get position, rotation and text for the hillas parameters label
    """
    try:
        length = hillas.length.to_value(unit)
        width = hillas.width.to_value(unit)
    except u.UnitsError:
        raise ValueError("hillas must be in same frame as geometry")

    if isinstance(hillas, HillasParametersContainer):
        cog_x = hillas.fov_lon.to_value(unit)
        cog_y = hillas.fov_lat.to_value(unit)
    else:
        raise TypeError(
            "hillas must be a HillasParametersContainer" f", got: {hillas} "
        )

    psi_rad = hillas.psi.to_value(u.rad)
    psi_deg = Angle(hillas.psi).wrap_at(180 * u.deg).to_value(u.deg)

    ret = dict(
        cog_x=float(cog_x),
        cog_y=float(cog_y),
        width=float(width),
        length=float(length),
        psi_rad=float(psi_rad),
        psi_uncert_rad=float(hillas.psi_uncertainty.to_value(u.rad)),
    )

    if not with_label:
        return ret

    # the following code dealing with x, y, angle
    # results in the minimal rotation of the text and puts the
    # label just outside the ellipse
    if psi_deg < -135:
        psi_deg += 180
        psi_rad += np.pi
    elif psi_deg > 135:
        psi_deg -= 180
        psi_rad -= np.pi

    if -45 < psi_deg <= 45:
        r = 1.2 * n_sigma * width
        label_x = cog_x + r * np.cos(psi_rad + 0.5 * np.pi)
        label_y = cog_y + r * np.sin(psi_rad + 0.5 * np.pi)
        rotation = psi_deg
    elif 45 < psi_deg <= 135:
        r = 1.2 * n_sigma * length
        label_x = cog_x + r * np.cos(psi_rad)
        label_y = cog_y + r * np.sin(psi_rad)
        rotation = psi_deg - 90
    else:
        r = 1.2 * n_sigma * length
        label_x = cog_x - r * np.cos(psi_rad)
        label_y = cog_y - r * np.sin(psi_rad)
        rotation = psi_deg + 90

    ret["rotation"] = float(rotation)
    ret["label_x"] = float(label_x)
    ret["label_y"] = float(label_y)

    if unit == u.deg:
        sep = ""
    else:
        sep = " "

    ret["text"] = (
        f"({cog_x:.2f}{sep}{unit:unicode}, {cog_y:.2f}{sep}{unit:unicode})\n"
        f"[w={width:.2f}{sep}{unit:unicode},l={length:.2f}{sep}{unit:unicode}]"
    )

    return ret
