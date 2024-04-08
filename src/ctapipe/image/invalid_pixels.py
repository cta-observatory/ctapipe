"""
Methods to interpolate broken pixels
"""

from abc import ABCMeta, abstractmethod

import numpy as np

from ..core import TelescopeComponent

__all__ = [
    "InvalidPixelHandler",
    "NeighborAverage",
]


class InvalidPixelHandler(TelescopeComponent, metaclass=ABCMeta):
    """
    An abstract base class for algorithms treating invalid pixel data in images
    """

    @abstractmethod
    def __call__(
        self, tel_id, image, peak_time, pixel_mask
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Handle invalid (broken, high noise, ...) pixels.

        E.g. replace by average over neighbors, see `NeighborAverage`).

        Parameters
        ----------
        tel_id : int
            telescope id
        image : np.ndarray
            Array of pixel image values
        peak_time : np.ndarray
            Array of pixel peak_time values
        pixel_mask : np.ndarray
            Boolean mask of the pixels to be interpolated
            Shape: (n_channels, n_pixels)

        Returns
        -------
        image : np.ndarray
            Image with interpolated values
        peak_time : np.ndarray
            peak_time with interpolated values
        """


class NeighborAverage(InvalidPixelHandler):
    def __call__(self, tel_id, image, peak_time, pixel_mask):
        """Interpolate pixels in dl1 images and peak_times

        Pixels to be interpolated are replaced by the average value of their neighbors.

        Pixels where no valid neighbors are available are filled with zeros.

        Parameters
        ----------
        tel_id : int
            telescope id
        image : np.ndarray
            Array of pixel image values
        peak_time : np.ndarray
            Array of pixel peak_time values
        pixel_mask : np.ndarray
            Boolean mask of the pixels to be interpolated
            Shape: (n_channels, n_pixels)

        Returns
        -------
        image : np.ndarray
            Image with interpolated values
        peak_time : np.ndarray
            peak_time with interpolated values
        """
        geometry = self.subarray.tel[tel_id].camera.geometry

        n_interpolated = np.count_nonzero(pixel_mask, axis=-1, keepdims=True)
        if (n_interpolated == 0).all():
            return image, peak_time

        image = np.atleast_2d(image)
        peak_time = np.atleast_2d(peak_time)
        for ichannel in range(image.shape[-2]):
            if n_interpolated[ichannel] == 0:
                continue
            # exclude to-be-interpolated pixels from neighbors
            neighbors = (
                geometry.neighbor_matrix[pixel_mask[ichannel]] & ~pixel_mask[ichannel]
            )

            index, neighbor = np.nonzero(neighbors)
            image_sum = np.zeros(n_interpolated[ichannel], dtype=image.dtype)
            count = np.zeros(n_interpolated[ichannel], dtype=int)
            peak_time_sum = np.zeros(n_interpolated[ichannel], dtype=peak_time.dtype)

            # calculate average of image and peak_time
            np.add.at(count, index, 1)
            np.add.at(image_sum, index, image[ichannel, neighbor])
            np.add.at(peak_time_sum, index, peak_time[ichannel, neighbor])

            valid = count > 0
            np.divide(image_sum, count, out=image_sum, where=valid)
            np.divide(peak_time_sum, count, out=peak_time_sum, where=valid)

            peak_time[ichannel, pixel_mask[ichannel]] = peak_time_sum
            image[ichannel, pixel_mask[ichannel]] = image_sum

        return np.squeeze(image), np.squeeze(peak_time)
