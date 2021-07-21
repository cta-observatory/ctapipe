import numpy as np
import astropy.units as u

__all__ = ["spread_line_seed", "create_seed", "rotate_translate",
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

    x_max_exp = 300 + 93 * np.log10(energy)

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
    centre_x, centre_y, amp = list(), list(), list()

    for tel_hillas in hillas:
        centre_x.append(tel_hillas.x.to(u.rad).value)
        centre_y.append(tel_hillas.y.to(u.rad).value)
        amp.append(tel_hillas.intensity)

    centre_x = np.average(centre_x, weights=amp)
    centre_y = np.average(centre_y, weights=amp)
    centre_tel_x = np.average(tel_x, weights=amp)
    centre_tel_y = np.average(tel_y, weights=amp)

    diff_x = source_x - centre_x
    diff_y = source_y - centre_y
    diff_tel_x = tilt_x - centre_tel_x
    diff_tel_y = tilt_y - centre_tel_y

    seed_list = list()

    for shift in shift_frac:
        seed_list.append(
            create_seed(
                centre_x + (diff_x * shift),
                centre_y + (diff_y * shift),
                centre_tel_x + (diff_tel_x * shift),
                centre_tel_y + (diff_tel_y * shift),
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
    lower_en_limit = energy * 0.5
    en_seed = energy

    # If our energy estimate falls outside of the range of our templates set it to
    # the edge
    if lower_en_limit < 0.01:
        lower_en_limit = 0.01
        en_seed = 0.01

    # Take the seed from Hillas-based reconstruction
    seed = (source_x, source_y, tilt_x, tilt_y, en_seed, 1)

    # Take a reasonable first guess at step size
    step = [0.04 / 57.3, 0.04 / 57.3, 5, 5, en_seed * 0.1, 0.05]
    # And some sensible limits of the fit range
    limits = [
        [source_x - 0.5/57.3, source_x + 0.5/57.3],
        [source_y - 0.5/57.3, source_y + 0.5/57.3],
        [tilt_x - 100, tilt_x + 100],
        [tilt_y - 100, tilt_y + 100],
        [lower_en_limit, en_seed * 2],
        [0.5, 2],
        [False, False]
    ]

    return seed, step, limits
