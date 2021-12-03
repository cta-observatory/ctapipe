import numpy as np
import astropy.units as u
from astropy.units import Quantity
from scipy.stats import norm 
from astropy.table import Table
from scipy.interpolate import interp1d

__all__ = ["spread_line_seed", "create_seed", "rotate_translate", "get_atmosphere_profile",
"guess_shower_depth", "energy_prior", "xmax_prior", "EmptyImages"]


class EmptyImages(Exception):
    pass


def guess_shower_depth(energy):
    """
    Simple estimation of depth of shower max based on the expected gamma-ray elongation
    rate.

    Parameters
    ----------
    energy: float
        Energy of the shower in TeV

    Returns
    -------
    float: Expected depth of shower maximum
    """

    x_max_exp = 300 + (93 * np.log10(energy))

    return x_max_exp


def energy_prior(energy, index=-1):
    return -2 * np.log(energy ** index)


def xmax_prior(energy, xmax, width=100):
    x_max_exp = guess_shower_depth(energy)
    diff = xmax - x_max_exp
    return -2 * np.log(norm.pdf(diff / width))


def rotate_translate(pixel_pos_x, pixel_pos_y, x_trans, y_trans, phi):
    """
    Function to perform rotation and translation of pixel lists

    Parameters
    ----------
    pixel_pos_x: ndarray
        Array of pixel x positions
    pixel_pos_y: ndarray
        Array of pixel x positions
    x_trans: float
        Translation of position in x coordinates
    y_trans: float
        Translation of position in y coordinates
    phi: float
        Rotation angle of pixels

    Returns
    -------
        ndarray,ndarray: Transformed pixel x and y coordinates

    """

    cosine_angle = np.cos(phi[..., np.newaxis])
    sin_angle = np.sin(phi[..., np.newaxis])

    pixel_pos_trans_x = (x_trans - pixel_pos_x) * cosine_angle - (
            y_trans - pixel_pos_y
    ) * sin_angle

    pixel_pos_trans_y = (pixel_pos_x - x_trans) * sin_angle + (
            pixel_pos_y - y_trans
    ) * cosine_angle
    return pixel_pos_trans_x, pixel_pos_trans_y


def spread_line_seed(
    hillas,
    tel_x,
    tel_y,
    source_x,
    source_y,
    tilt_x,
    tilt_y,
    energy,
    shift_frac=[2, 1.5, 1, 0.5, 0, -0.5, -1, -1.5],
):
    """
    Parameters
    ----------
    hillas: list
        Hillas parameters in event
    tel_x: list
        telescope X positions in tilted system
    tel_y: list
        telescope Y positions in tilted system
    source_x: float
        Source X position in nominal system (radians)
    source_y:float
        Source Y position in nominal system (radians)
    tilt_x: float
        Core X position in tilited system (radians)
    tilt_y: float
        Core Y position in tilited system (radians)
    energy: float
        Energy in TeV
    shift_frac: list
        Fractional values to shist source and core positions

    Returns
    -------
    list of seed positions to try
    """

    seed_list = list()

    seed_list.append(
        create_seed(
            source_x,
            source_y,
            tilt_x,
            tilt_y,
            energy,
        )
    )

    return seed_list


def create_seed(source_x, source_y, tilt_x, tilt_y, energy):
    """
    Function for creating seed, step and limits for a given position

    Parameters
    ----------
    source_x: float
        Source X position in nominal system (radians)
    source_y:float
        Source Y position in nominal system (radians)
    tilt_x: float
        Core X position in tilited system (radians)
    tilt_y: float
        Core Y position in tilited system (radians)
    energy: float
        Energy in TeV

    Returns
    -------
    tuple of seed, steps size and fit limits
    """
    lower_en_limit = energy * 0.1
    en_seed = energy

    # If our energy estimate falls outside of the range of our templates set it to
    # the edge
    if lower_en_limit < 0.01:
        lower_en_limit = 0.01
        en_seed = 0.01

    # Take the seed from Hillas-based reconstruction
    seed = (source_x, source_y, tilt_x, tilt_y, en_seed, 1.)

    # Take a reasonable first guess at step size
    step = [0.0001 / 57.3, 0.0001 / 57.3, 10, 10, en_seed * 0.05, 0.05, 0.]
    # And some sensible limits of the fit range
    limits = [
        [source_x - 0.5/57.3, source_x + 0.5/57.3],
        [source_y - 0.5/57.3, source_y + 0.5/57.3],
        [tilt_x - 100, tilt_x + 100],
        [tilt_y - 100, tilt_y + 100],
        [lower_en_limit, en_seed * 2],
        [0.8, 1.2],
        [0.0, 0.01]
    ]

    return seed, step, limits

def get_atmosphere_profile(filename, with_units=True):
    """
    Gives atmospheric profile as a continuous function thickness(
    altitude), and it's inverse altitude(thickness)  in m and g/cm^2
    respectively.

    Parameters
    ----------
    atmosphere_name: str
        identifier of atmosphere profile
    with_units: bool
       if true, return functions that accept and return unit quantities.
       Otherwise assume units are 'm' and 'g cm-2'

    Returns
    -------
    functions: thickness(alt), alt(thickness)
    """

    data = Table()
    
    tab = data.read(filename)
    alt = tab["altitude"].to("m")
    thick = (tab["thickness"]).to("g cm-2")

    alt_to_thickness = interp1d(x=np.array(alt), y=np.array(thick))
    thickness_to_alt = interp1d(x=np.array(thick), y=np.array(alt))

    if with_units:

        def thickness(a):
            return Quantity(alt_to_thickness(a.to("m")), "g cm-2")

        def altitude(a):
            return Quantity(thickness_to_alt(a.to("g cm-2")), "m")

        return thickness, altitude

    return alt_to_thickness, thickness_to_alt

