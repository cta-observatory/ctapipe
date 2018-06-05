__author__ = "@jeremiedecock"

import numpy as np


def astri_transformation_map():
    """Make the transformation map to turn the 1D array of an ASTRI image into a
    rectangular 2D array. This will add new "fake" pixels that are filled with NaN value.
    """

    # By default, pixels map to the last element of input_img_ext (i.e. NaN)
    img_map = np.full([56, 56], -1, dtype=int)

    # Map values
    img_map[0:8, 16:24] = np.arange(64).reshape([8, 8])[::-1, :] + 34 * 64
    img_map[0:8, 24:32] = np.arange(64).reshape([8, 8])[::-1, :] + 35 * 64
    img_map[0:8, 32:40] = np.arange(64).reshape([8, 8])[::-1, :] + 36 * 64

    img_map[8:16, 8:16] = np.arange(64).reshape([8, 8])[::-1, :] + 29 * 64
    img_map[8:16, 16:24] = np.arange(64).reshape([8, 8])[::-1, :] + 30 * 64
    img_map[8:16, 24:32] = np.arange(64).reshape([8, 8])[::-1, :] + 31 * 64
    img_map[8:16, 32:40] = np.arange(64).reshape([8, 8])[::-1, :] + 32 * 64
    img_map[8:16, 40:48] = np.arange(64).reshape([8, 8])[::-1, :] + 33 * 64

    img_map[16:24, 0:8] = np.arange(64).reshape([8, 8])[::-1, :] + 22 * 64
    img_map[16:24, 8:16] = np.arange(64).reshape([8, 8])[::-1, :] + 23 * 64
    img_map[16:24, 16:24] = np.arange(64).reshape([8, 8])[::-1, :] + 24 * 64
    img_map[16:24, 24:32] = np.arange(64).reshape([8, 8])[::-1, :] + 25 * 64
    img_map[16:24, 32:40] = np.arange(64).reshape([8, 8])[::-1, :] + 26 * 64
    img_map[16:24, 40:48] = np.arange(64).reshape([8, 8])[::-1, :] + 27 * 64
    img_map[16:24, 48:56] = np.arange(64).reshape([8, 8])[::-1, :] + 28 * 64

    img_map[24:32, 0:8] = np.arange(64).reshape([8, 8])[::-1, :] + 15 * 64
    img_map[24:32, 8:16] = np.arange(64).reshape([8, 8])[::-1, :] + 16 * 64
    img_map[24:32, 16:24] = np.arange(64).reshape([8, 8])[::-1, :] + 17 * 64
    img_map[24:32, 24:32] = np.arange(64).reshape([8, 8])[::-1, :] + 18 * 64
    img_map[24:32, 32:40] = np.arange(64).reshape([8, 8])[::-1, :] + 19 * 64
    img_map[24:32, 40:48] = np.arange(64).reshape([8, 8])[::-1, :] + 20 * 64
    img_map[24:32, 48:56] = np.arange(64).reshape([8, 8])[::-1, :] + 21 * 64

    img_map[32:40, 0:8] = np.arange(64).reshape([8, 8])[::-1, :] + 8 * 64
    img_map[32:40, 8:16] = np.arange(64).reshape([8, 8])[::-1, :] + 9 * 64
    img_map[32:40, 16:24] = np.arange(64).reshape([8, 8])[::-1, :] + 10 * 64
    img_map[32:40, 24:32] = np.arange(64).reshape([8, 8])[::-1, :] + 11 * 64
    img_map[32:40, 32:40] = np.arange(64).reshape([8, 8])[::-1, :] + 12 * 64
    img_map[32:40, 40:48] = np.arange(64).reshape([8, 8])[::-1, :] + 13 * 64
    img_map[32:40, 48:56] = np.arange(64).reshape([8, 8])[::-1, :] + 14 * 64

    img_map[40:48, 8:16] = np.arange(64).reshape([8, 8])[::-1, :] + 3 * 64
    img_map[40:48, 16:24] = np.arange(64).reshape([8, 8])[::-1, :] + 4 * 64
    img_map[40:48, 24:32] = np.arange(64).reshape([8, 8])[::-1, :] + 5 * 64
    img_map[40:48, 32:40] = np.arange(64).reshape([8, 8])[::-1, :] + 6 * 64
    img_map[40:48, 40:48] = np.arange(64).reshape([8, 8])[::-1, :] + 7 * 64

    img_map[48:56, 16:24] = np.arange(64).reshape([8, 8])[::-1, :] + 0 * 64
    img_map[48:56, 24:32] = np.arange(64).reshape([8, 8])[::-1, :] + 1 * 64
    img_map[48:56, 32:40] = np.arange(64).reshape([8, 8])[::-1, :] + 2 * 64

    return img_map


def astri_to_2d_array_no_crop(input_img, img_map=astri_transformation_map()):
    """Convert images comming form "ASTRI" telescopes in order to get regular 2D
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
    if len(input_img) != (37 * 64):
        raise ValueError("The input image is not a valide ASTRI telescope image.")

    # Copy the input flat ctapipe image and add one element with the NaN value in the end

    input_img_ext = np.zeros(input_img.shape[0] + 1)
    input_img_ext[:-1] = input_img[:]
    input_img_ext[-1] = np.nan

    # Make the output image
    img_2d = input_img_ext[img_map]

    return img_2d


# for archaic consistency; there was another version of the function that would cut
# off over-standing parts of the image
astri_to_2d_array = astri_to_2d_array_no_crop


def array_2d_to_astri(img_2d):
    """Transforms the 2D ASTRI image back to 1D

    Parameters
    ----------
    img_2d : 2d numpy array
        the 2D ASTRI image

    Returns
    -------
    img_1d : 1d numpy array
        the 1D ASTRI image
    """

    img_1d = np.concatenate([
        img_2d[48:56, 16:24][::-1, :].ravel(),
        img_2d[48:56, 24:32][::-1, :].ravel(),
        img_2d[48:56, 32:40][::-1, :].ravel(),
        #
        img_2d[40:48, 8:16][::-1, :].ravel(),
        img_2d[40:48, 16:24][::-1, :].ravel(),
        img_2d[40:48, 24:32][::-1, :].ravel(),
        img_2d[40:48, 32:40][::-1, :].ravel(),
        img_2d[40:48, 40:48][::-1, :].ravel(),
        #
        img_2d[32:40, 0:8][::-1, :].ravel(),
        img_2d[32:40, 8:16][::-1, :].ravel(),
        img_2d[32:40, 16:24][::-1, :].ravel(),
        img_2d[32:40, 24:32][::-1, :].ravel(),
        img_2d[32:40, 32:40][::-1, :].ravel(),
        img_2d[32:40, 40:48][::-1, :].ravel(),
        img_2d[32:40, 48:56][::-1, :].ravel(),
        #
        img_2d[24:32, 0:8][::-1, :].ravel(),
        img_2d[24:32, 8:16][::-1, :].ravel(),
        img_2d[24:32, 16:24][::-1, :].ravel(),
        img_2d[24:32, 24:32][::-1, :].ravel(),
        img_2d[24:32, 32:40][::-1, :].ravel(),
        img_2d[24:32, 40:48][::-1, :].ravel(),
        img_2d[24:32, 48:56][::-1, :].ravel(),
        #
        img_2d[16:24, 0:8][::-1, :].ravel(),
        img_2d[16:24, 8:16][::-1, :].ravel(),
        img_2d[16:24, 16:24][::-1, :].ravel(),
        img_2d[16:24, 24:32][::-1, :].ravel(),
        img_2d[16:24, 32:40][::-1, :].ravel(),
        img_2d[16:24, 40:48][::-1, :].ravel(),
        img_2d[16:24, 48:56][::-1, :].ravel(),
        #
        img_2d[8:16, 8:16][::-1, :].ravel(),
        img_2d[8:16, 16:24][::-1, :].ravel(),
        img_2d[8:16, 24:32][::-1, :].ravel(),
        img_2d[8:16, 32:40][::-1, :].ravel(),
        img_2d[8:16, 40:48][::-1, :].ravel(),
        #
        img_2d[0:8, 16:24][::-1, :].ravel(),
        img_2d[0:8, 24:32][::-1, :].ravel(),
        img_2d[0:8, 32:40][::-1, :].ravel()
    ])

    return img_1d
