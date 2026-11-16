import gzip
import pickle

import numba
import numpy as np
from scipy.stats import norm

__all__ = [
    "create_seed",
    "rotate_translate",
    "generate_fake_template",
    "create_dummy_templates",
    "guess_shower_depth",
    "EmptyImages",
]


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


@numba.njit
def rotate_translate(pixel_pos_x, pixel_pos_y, x_trans, y_trans, phi):
    """
    Function to perform rotation and translation of pixel lists. Array
    manipulation slowing this function significantly. Now Numba accelerated.

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
    shape = pixel_pos_x.shape
    pixel_pos_trans_x, pixel_pos_trans_y = np.zeros(shape), np.zeros(shape)

    for i in range(shape[0]):
        cosine_angle = np.cos(phi[i])
        sin_angle = np.sin(phi[i])

        for j in range(shape[1]):
            pixel_pos_trans_x[i][j] = (x_trans - pixel_pos_x[i][j]) * cosine_angle - (
                y_trans - pixel_pos_y[i][j]
            ) * sin_angle
            pixel_pos_trans_y[i][j] = (pixel_pos_x[i][j] - x_trans) * sin_angle + (
                pixel_pos_y[i][j] - y_trans
            ) * cosine_angle

    return pixel_pos_trans_x, pixel_pos_trans_y


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
    if lower_en_limit < 0.02:
        lower_en_limit = 0.02

    # Take the seed from Hillas-based reconstruction
    seed = [source_x, source_y, tilt_x, tilt_y, en_seed, 1.0]
    if energy > 2:
        seed = [source_x, source_y, tilt_x, tilt_y, en_seed, 1.2]

    # Take a reasonable first guess at step size
    step = [0.04 / 57.3, 0.04 / 57.3, 10, 10, en_seed * 0.05, 0.05, 0.01]
    # And some sensible limits of the fit range
    limits = [
        [source_x - 1.5 / 57.3, source_x + 1.5 / 57.3],
        [source_y - 1.5 / 57.3, source_y + 1.5 / 57.3],
        [tilt_x - 100, tilt_x + 100],
        [tilt_y - 100, tilt_y + 100],
        [lower_en_limit, en_seed * 2],
        [0.8, 1.2],
        [0.0, 0.01],
    ]

    return seed, step, limits


def generate_fake_template(
    center, length, width=0.3, xb=301, yb=151, bounds=((-5, 1), (-1.5, 1.5))
):
    """Simple function to generate template for testing

    Args:
        center (float): X axis coordinate of image center
        length (float): Image length
        width (float, optional): Image width. Defaults to 0.3.
        xb (int, optional): Number of x bins in template. Defaults to 300.
        yb (int, optional): Number of y bins in template. Defaults to 150.
        bounds (tuple, optional): Boundaries of templates. Defaults to ((-5, 1), (-1.5, 1.5)).

    Returns:
        ndarray, ndarray, ndarray: Template image, x bin centres, y bin centres
    """
    x, y = np.meshgrid(
        np.linspace(bounds[0][0], bounds[0][1], xb),
        np.linspace(bounds[1][0], bounds[1][1], yb),
    )

    template = np.zeros((yb, xb), "float")
    selection = np.abs(y) < width
    selection = np.logical_and(selection, np.abs(x - center) < length)

    template[selection] = 1

    return template, x, y


def create_dummy_templates(
    output_file,
    energy,
    pe=1000.0,
    energy_range=np.logspace(-1, 1, 7),
    dist_range=np.linspace(0, 200, 5),
    xmax_range=np.linspace(-200, 200, 9),
):
    """Create file with dummy template library

    Args:
        output_file (str): Output file name
        energy (float): Peak energy of templates
        pe (float, optional): Peak template amplitude. Defaults to 1000.
        energy_range (ndarray, optional): Range of energy templates. Defaults to np.logspace(-1, 1,7).
        dist_range (ndarray, optional): Range of distance templates. Defaults to np.linspace(0, 200, 5).
        xmax_range (ndarray, optional): Range of xmax templates. Defaults to np.linspace(-200,200, 9).
    """
    template_dict = {}
    for en in energy_range:
        scale = norm.pdf(np.log10(en), loc=np.log10(energy), scale=0.3) / norm.pdf(
            np.log10(energy), loc=np.log10(energy), scale=0.3
        )
        for dist in dist_range:
            for xmax in xmax_range:
                key = (0, 0, en, dist, xmax)
                template, x, y = generate_fake_template(-1.5, 0.5)
                template *= scale

                template_dict[key] = template.T * pe

    with gzip.open(output_file, "wb") as filehandler:
        pickle.dump(template_dict, filehandler)
