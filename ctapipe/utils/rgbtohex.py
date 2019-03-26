from matplotlib.cm import get_cmap
import numpy as np
import codecs
viridis = get_cmap('viridis')


def intensity_to_rgb(array, minval=None, maxval=None):
    """
    Converts the values of an array to rgb representing a color for a z axis

    Parameters
    ----------
    array : ndarray
        1D numpy array containing intensity values for a z axis
    minval: int
        minimum value of the image
    maxval: int
        maximum value of the image

    Returns
    -------
    rgb : ndarray
        rgb tuple representing the intensity as a color

    """
    if minval is None:
        minval = array.min()
    if maxval is None:
        maxval = array.max()
    if maxval == minval:
        minval -= 1
        maxval += 1
    scaled = (array - minval) / (maxval - minval)

    rgb = (255 * viridis(scaled)).astype(np.uint8)
    return rgb


def intensity_to_hex(array, minval=None, maxval=None):
    """
    Converts the values of an array to hex representing a color for a z axis

    This is needed to efficiently change the values displayed by a
    `ctapipe.visualization.bokeh.CameraDisplay`.

    Parameters
    ----------
    array : ndarray
        1D numpy array containing intensity values for a z axis
    minval: int
        minimum value of the image
    maxval: int
        maximum value of the image

    Returns
    -------
    hex_ : ndarray
        hex strings representing the intensity as a color

    """
    hex_ = np.zeros((array.size, 8), dtype='B')
    rgb = intensity_to_rgb(array, minval, maxval)

    hex_encoded = codecs.encode(rgb, 'hex')
    bytes_ = np.frombuffer(hex_encoded, 'B')
    bytes_2d = bytes_.reshape(-1, 8)
    hex_[:, 0] = ord('#')
    hex_[:, 1:7] = bytes_2d[:, 0:6]

    return hex_.view('S8').astype('U8')[:, 0]
