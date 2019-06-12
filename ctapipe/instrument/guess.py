"""
Methods for guessing details about a telescope from some metadata (so that
we can create TelescopeDescriptions from Monte-Carlo where some
parameters like the names of the camera and optics structure are not
stored in the file.
"""
from collections import namedtuple

import astropy.units as u

GuessingKey = namedtuple("GuessingKey", ["n_pixels", "focal_length"])
GuessingResult = namedtuple(
    "GuessingResult", ["type", "name", "camera_name", "n_mirrors"]
)


TELESCOPE_NAMES = {
    GuessingKey(2048, 2.28): GuessingResult("SST", "GCT", "CHEC", 2),
    GuessingKey(2368, 2.15): GuessingResult("SST", "ASTRI", "ASTRICam", 2),
    GuessingKey(2048, 2.15): GuessingResult("SST", "ASTRI", "CHEC", 2),
    GuessingKey(1296, 5.60): GuessingResult("SST", "1M", "DigiCam", 1),
    GuessingKey(1764, 16.0): GuessingResult("MST", "MST", "FlashCam", 1),
    GuessingKey(1855, 16.0): GuessingResult("MST", "MST", "NectarCam", 1),
    GuessingKey(1855, 28.0): GuessingResult("LST", "LST", "LSTCam", 1),
    GuessingKey(11328, 5.59): GuessingResult("MST", "SCT", "SCTCam", 1),
    # Non-CTA Telescopes
    GuessingKey(960, 15.0): GuessingResult("MST", "HESS-I", "HESS-I", 1),
    GuessingKey(2048, 36.0): GuessingResult("LST", "HESS-II", "HESS-II", 1),
    GuessingKey(1440, 4.998): GuessingResult("SST", "FACT", "FACT", 1),
}

UNKNOWN_TELESCOPE = GuessingResult("UNKNOWN", "UNKNOWN", "UNKNOWN", -1)


def guess_telescope(n_pixels, focal_length):
    """
    From n_pixels of the camera and the focal_length,
    guess which telescope we are dealing with.
    This is mainly needed to add human readable names
    to telescopes read from simtel array.

    Parameters
    ----------
    n_pixels: int
        number of pixels of the telescope's camera
    focal_length: float or u.Quantity[length]
        Focal length, either in m or as astropy quantity

    Returns
    -------
    result: GuessingResult
        A namedtuple having type, telescope_name, camera_name and n_mirrors fields
    """

    # allow unit input
    focal_length = u.Quantity(focal_length, u.m).to_value(u.m)

    try:
        return TELESCOPE_NAMES[(n_pixels, round(focal_length, 2))]
    except KeyError:
        raise ValueError(f"Unknown telescope: n_pixel={n_pixels}, f={focal_length}")
