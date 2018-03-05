__author__ = "@jeremiedecock"

import numpy as np


def chec_transformation_map():
    # By default, pixels map to the last element of input_img_ext (i.e. NaN)
    img_map = np.full([8 * 6, 8 * 6], -1, dtype=int)

    # Map values
    img_map[:8, 8:-8] = np.arange(8 * 8 * 4).reshape([8, 8 * 4])
    img_map[8:40, :] = np.arange(32 * 48).reshape([32, 48]) + 256
    img_map[-8:, 8:-8] = np.arange(8 * 8 * 4).reshape([8, 8 * 4]) + 1792

    return img_map


def chec_to_2d_array(input_img, img_map=chec_transformation_map()):
    """
    Convert images comming form "CHEC" cameras in order to get regular 2D
    "rectangular" images directly usable with most image processing tools.

    Parameters
    ----------
    input_img : numpy.array
        The image to convert

    Returns
    -------
    A numpy.array containing the cropped image.
    """

    # Check the image
    if len(input_img) != 2048:
        raise ValueError("The input image is not a valide CHEC camera image.")

    # Copy the input flat ctapipe image and add one element with the NaN value in the end

    input_img_ext = np.zeros(input_img.shape[0] + 1)
    input_img_ext[:-1] = input_img[:]
    input_img_ext[-1] = np.nan

    # Make the output image
    img_2d = input_img_ext[img_map]

    return img_2d


def array_2d_to_chec(img_2d):
    # Flatten image and remove NaN values
    img_1d = img_2d[np.isfinite(img_2d)]

    return img_1d
