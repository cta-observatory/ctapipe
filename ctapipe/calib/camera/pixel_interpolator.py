"""
Methods to interpolate broken pixels
"""
import numpy as np

__all__ = [
    "interpolate_pixels",
]


def interpolate_pixels(image, peak_time, pixel_mask, geometry):
    """Interpolate pixels in dl1 images and peak_times

    Pixels to be interpolated are replaced by the average value their neighbors.

    Pixels where no valid neighbors are available are filled with zeros.

    Parameters
    ----------
    image : np.ndarray
        Array of pixel image values
    peak_time : np.ndarray
        Array of pixel peak_time values
    pixel_mask : np.ndarray
        Boolean mask of the pixels to be interpolated
    geometry : ctapipe.instrument.CameraGeometry
        camera geometry corresponding to the image

    Returns
    -------
    image : np.ndarray
        Image with interpolated values
    peak_time : np.ndarray
        peak_time with interpolated values
    """

    n_interpolated = np.count_nonzero(pixel_mask)
    if n_interpolated == 0:
        return image, peak_time

    # exclude to-be-interpolated pixels from neighbors
    neighbors = geometry.neighbor_matrix[pixel_mask] & ~pixel_mask

    index, neighbor = np.nonzero(neighbors)
    image_sum = np.zeros(n_interpolated, dtype=image.dtype)
    count = np.zeros(n_interpolated, dtype=int)
    peak_time_sum = np.zeros(n_interpolated, dtype=peak_time.dtype)

    # calculate average of image and peak_time
    np.add.at(count, index, 1)
    np.add.at(image_sum, index, image[neighbor])
    np.add.at(peak_time_sum, index, peak_time[neighbor])

    valid = count > 0
    np.divide(image_sum, count, out=image_sum, where=valid)
    np.divide(peak_time_sum, count, out=peak_time_sum, where=valid)

    peak_time = peak_time.copy()
    peak_time[pixel_mask] = peak_time_sum

    image = image.copy()
    image[pixel_mask] = image_sum

    return image, peak_time
