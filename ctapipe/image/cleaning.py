"""
Image Cleaning Algorithms (identification of noisy pixels)
"""

__all__ = [
    "tailcuts_clean",
    "dilate",
    "mars_cleaning_1st_pass",
    "fact_image_cleaning",
    "apply_time_delta_cleaning",
    "ImageCleaner",
    "TailcutsImageCleaner",
]

from abc import abstractmethod

import numpy as np

from ..core.component import TelescopeComponent
from ..core.traits import (
    FloatTelescopeParameter,
    IntTelescopeParameter,
    BoolTelescopeParameter,
)


def tailcuts_clean(
    geom,
    image,
    picture_thresh=7,
    boundary_thresh=5,
    keep_isolated_pixels=False,
    min_number_picture_neighbors=0,
):

    """Clean an image by selection pixels that pass a two-threshold
    tail-cuts procedure.  The picture and boundary thresholds are
    defined with respect to the pedestal dispersion. All pixels that
    have a signal higher than the picture threshold will be retained,
    along with all those above the boundary threshold that are
    neighbors of a picture pixel.

    To include extra neighbor rows of pixels beyond what are accepted, use the
    `ctapipe.image.dilate` function.

    Parameters
    ----------
    geom: `ctapipe.instrument.CameraGeometry`
        Camera geometry information
    image: array
        pixel values
    picture_thresh: float or array
        threshold above which all pixels are retained
    boundary_thresh: float or array
        threshold above which pixels are retained if they have a neighbor
        already above the picture_thresh
    keep_isolated_pixels: bool
        If True, pixels above the picture threshold will be included always,
        if not they are only included if a neighbor is in the picture or
        boundary
    min_number_picture_neighbors: int
        A picture pixel survives cleaning only if it has at least this number
        of picture neighbors. This has no effect in case keep_isolated_pixels is True

    Returns
    -------

    A boolean mask of *clean* pixels.  To get a zero-suppressed image and pixel
    list, use `image[mask], geom.pix_id[mask]`, or to keep the same
    image size and just set unclean pixels to 0 or similar, use
    `image[~mask] = 0`

    """
    pixels_above_picture = image >= picture_thresh

    if keep_isolated_pixels or min_number_picture_neighbors == 0:
        pixels_in_picture = pixels_above_picture
    else:
        # Require at least min_number_picture_neighbors. Otherwise, the pixel
        #  is not selected
        number_of_neighbors_above_picture = geom.neighbor_matrix_sparse.dot(
            pixels_above_picture.view(np.byte)
        )
        pixels_in_picture = pixels_above_picture & (
            number_of_neighbors_above_picture >= min_number_picture_neighbors
        )

    # by broadcasting together pixels_in_picture (1d) with the neighbor
    # matrix (2d), we find all pixels that are above the boundary threshold
    # AND have any neighbor that is in the picture
    pixels_above_boundary = image >= boundary_thresh
    pixels_with_picture_neighbors = geom.neighbor_matrix_sparse.dot(pixels_in_picture)
    if keep_isolated_pixels:
        return (
            pixels_above_boundary & pixels_with_picture_neighbors
        ) | pixels_in_picture
    else:
        pixels_with_boundary_neighbors = geom.neighbor_matrix_sparse.dot(
            pixels_above_boundary
        )
        return (pixels_above_boundary & pixels_with_picture_neighbors) | (
            pixels_in_picture & pixels_with_boundary_neighbors
        )


def mars_cleaning_1st_pass(
    geom,
    image,
    picture_thresh=7,
    boundary_thresh=5,
    keep_isolated_pixels=False,
    min_number_picture_neighbors=0,
):
    """
    Clean an image by selection pixels that pass a three-threshold tail-cuts
    procedure.
    All thresholds are defined with respect to the pedestal
    dispersion. All pixels that have a signal higher than the core (picture)
    threshold will be retained, along with all those above the boundary
    threshold that are neighbors of a core pixel AND all those above
    the boundary threshold that are neighbors of a neighbor of a core pixel.

    Parameters
    ----------
    geom: `ctapipe.instrument.CameraGeometry`
        Camera geometry information
    image: array
        pixel values
    picture_thresh: float
        threshold above which all pixels are retained
    boundary_thresh: float
        threshold above which pixels are retained if
        they have a neighbor already above the picture_thresh; it is then
        reapplied iteratively to the neighbor of the neighbor
    keep_isolated_pixels: bool
        If True, pixels above the picture threshold will be included always,
        if not they are only included if a neighbor is in the picture or
        boundary
    min_number_picture_neighbors: int
        A picture pixel survives cleaning only if it has at least this number
        of picture neighbors. This has no effect in case keep_isolated_pixels is True

    Returns
    -------
    A boolean mask of *clean* pixels.  To get a zero-suppressed image and pixel
    list, use `image[mask], geom.pix_id[mask]`, or to keep the same
    image size and just set unclean pixels to 0 or similar, use
    `image[~mask] = 0`

    """

    pixels_from_tailcuts_clean = tailcuts_clean(
        geom,
        image,
        picture_thresh,
        boundary_thresh,
        keep_isolated_pixels,
        min_number_picture_neighbors,
    )  # this selects any core pixel and any of its first neighbors

    # At this point we don't know yet which ones should be kept.
    # In principle, the pixel thresholds should be hierarchical from core to
    # boundaries (this should be true for every type of particle triggering
    # the image), so we can just check which pixels have more than
    # boundary_thresh photo-electrons in the same image, but starting from
    # the mask we got from 'tailcuts_clean'.

    pixels_above_2nd_boundary = image >= boundary_thresh

    # and now it's the same as the last part of 'tailcuts_clean', but without
    # the core pixels, i.e. we start from the neighbors of the core pixels.
    pixels_with_previous_neighbors = geom.neighbor_matrix_sparse.dot(
        pixels_from_tailcuts_clean
    )
    if keep_isolated_pixels:
        return (
            pixels_above_2nd_boundary & pixels_with_previous_neighbors
        ) | pixels_from_tailcuts_clean
    else:
        pixels_with_2ndboundary_neighbors = geom.neighbor_matrix_sparse.dot(
            pixels_above_2nd_boundary
        )
        return (pixels_above_2nd_boundary & pixels_with_previous_neighbors) | (
            pixels_from_tailcuts_clean & pixels_with_2ndboundary_neighbors
        )


def dilate(geom, mask):
    """
    Add one row of neighbors to the True values of a pixel mask and return
    the new mask.
    This can be used to include extra rows of pixels in a mask that was
    pre-computed, e.g. via `tailcuts_clean`.

    Parameters
    ----------
    geom: `~ctapipe.instrument.CameraGeometry`
        Camera geometry information
    mask: ndarray
        input mask (array of booleans) to be dilated
    """
    return mask | geom.neighbor_matrix_sparse.dot(mask)


def apply_time_delta_cleaning(
    geom, mask, arrival_times, min_number_neighbors, time_limit
):
    """
    Identify all pixels from selection that have less than N
    neighbors that arrived within a given timeframe.

    Parameters
    ----------
    geom: `ctapipe.instrument.CameraGeometry`
        Camera geometry information
    mask: array, boolean
        boolean mask of *clean* pixels before time_delta_cleaning
    arrival_times: array
        pixel timing information
    min_number_neighbors: int
        Threshold to determine if a pixel survives cleaning steps.
        These steps include checks of neighbor arrival time and value
    time_limit: int or float
        arrival time limit for neighboring pixels

    Returns
    -------

    A boolean mask of *clean* pixels.  To get a zero-suppressed image and pixel
    list, use `image[mask], geom.pix_id[mask]`, or to keep the same
    image size and just set unclean pixels to 0 or similar, use
    `image[~mask] = 0`

    """
    pixels_to_remove = []
    mask = mask.copy()  # Create copy so orginal is unchanged
    for pixel in np.where(mask)[0]:
        neighbors = geom.neighbor_matrix_sparse[pixel].indices
        time_diff = np.abs(arrival_times[neighbors] - arrival_times[pixel])
        if sum(time_diff < time_limit) < min_number_neighbors:
            pixels_to_remove.append(pixel)
    mask[pixels_to_remove] = False
    return mask


def fact_image_cleaning(
    geom,
    image,
    arrival_times,
    picture_threshold=4,
    boundary_threshold=2,
    min_number_neighbors=2,
    time_limit=5,
):
    """Clean an image by selection pixels that pass the fact cleaning procedure.

    Cleaning contains the following steps:
    1: Find pixels containing more photons than a threshold t1
    2: Remove pixels with less than N neighbors
    3: Add neighbors of the remaining pixels that are above a lower threshold t2
    4: Remove pixels with less than N neighbors arriving within a given timeframe
    5: Remove pixels with less than N neighbors
    6: Remove pixels with less than N neighbors arriving within a given timeframe

    Parameters
    ----------
    geom: `ctapipe.instrument.CameraGeometry`
        Camera geometry information
    image: array
        pixel values
    arrival_times: array
        pixel timing information
    picture_threshold: float or array
        threshold above which all pixels are retained
    boundary_threshold: float or array
        threshold above which pixels are retained if they have a neighbor
        already above the picture_thresh
    min_number_neighbors: int
        Threshold to determine if a pixel survives cleaning steps.
        These steps include checks of neighbor arrival time and value
    time_limit: int or float
        arrival time limit for neighboring pixels

    Returns
    -------
    A boolean mask of *clean* pixels.  To get a zero-suppressed image and pixel
    list, use `image[mask], geom.pix_id[mask]`, or to keep the same
    image size and just set unclean pixels to 0 or similar, use
    `image[~mask] = 0`

    References
    ----------
    See  [temme2016]_ and for implementation [factcleaning]_

    """

    # Step 1
    pixels_to_keep = image >= picture_threshold

    # Step 2
    number_of_neighbors_above_picture = geom.neighbor_matrix_sparse.dot(
        (pixels_to_keep).view(np.byte)
    )
    pixels_to_keep = pixels_to_keep & (
        number_of_neighbors_above_picture >= min_number_neighbors
    )

    # Step 3
    pixels_above_boundary = image >= boundary_threshold
    pixels_to_keep = dilate(geom, pixels_to_keep) & pixels_above_boundary

    # nothing else to do if min_number_neighbors <= 0
    if min_number_neighbors <= 0:
        return pixels_to_keep

    # Step 4
    pixels_to_keep = apply_time_delta_cleaning(
        geom, pixels_to_keep, arrival_times, min_number_neighbors, time_limit
    )

    # Step 5
    number_of_neighbors = geom.neighbor_matrix_sparse.dot(
        (pixels_to_keep).view(np.byte)
    )
    pixels_to_keep = pixels_to_keep & (number_of_neighbors >= min_number_neighbors)

    # Step 6
    pixels_to_keep = apply_time_delta_cleaning(
        geom, pixels_to_keep, arrival_times, min_number_neighbors, time_limit
    )
    return pixels_to_keep


class ImageCleaner(TelescopeComponent):
    """
    Abstract class for all configurable Image Cleaning algorithms.   Use
    `ImageCleaner.from_name()` to construct an instance of a particular algorithm
    """

    @abstractmethod
    def __call__(
        self, tel_id: int, image: np.ndarray, arrival_times: np.ndarray = None
    ) -> np.ndarray:
        """
        Identify pixels with signal, and reject those with pure noise.

        Parameters
        ----------
        tel_id: int
            which telescope id in the subarray is being used (determines
            which cut is used)
        image : np.ndarray
            image pixel data corresponding to the camera geometry
        arrival_times: np.ndarray
            image of arrival time (not used in this method)

        Returns
        -------
        np.ndarray
            boolean mask of pixels passing cleaning
        """
        pass


class TailcutsImageCleaner(ImageCleaner):
    """
    Clean images using the standard picture/boundary technique. See
    `ctapipe.image.tailcuts_clean`
    """

    picture_threshold_pe = FloatTelescopeParameter(
        default_value=10.0, help="top-level threshold in photoelectrons"
    ).tag(config=True)

    boundary_threshold_pe = FloatTelescopeParameter(
        default_value=5.0, help="second-level threshold in photoelectrons"
    ).tag(config=True)

    min_picture_neighbors = IntTelescopeParameter(
        default_value=2, help="Minimum number of neighbors above threshold to consider"
    ).tag(config=True)

    keep_isolated_pixels = BoolTelescopeParameter(
        default_value=False,
        help="If False, pixels with less neighbors than ``min_picture_neighbors`` are"
        "removed.",
    ).tag(config=True)

    def __call__(
        self, tel_id: int, image: np.ndarray, arrival_times=None
    ) -> np.ndarray:
        """
        Apply standard picture-boundary cleaning. See `ImageCleaner.__call__()`
        """

        return tailcuts_clean(
            self.subarray.tel[tel_id].camera.geometry,
            image,
            picture_thresh=self.picture_threshold_pe.tel[tel_id],
            boundary_thresh=self.boundary_threshold_pe.tel[tel_id],
            min_number_picture_neighbors=self.min_picture_neighbors.tel[tel_id],
            keep_isolated_pixels=self.keep_isolated_pixels.tel[tel_id],
        )


class MARSImageCleaner(TailcutsImageCleaner):
    """
    1st-pass MARS-like Image cleaner (See `ctapipe.image.mars_cleaning_1st_pass`)
    """

    def __call__(
        self, tel_id: int, image: np.ndarray, arrival_times=None
    ) -> np.ndarray:
        """
        Apply MARS-style image cleaning. See `ImageCleaner.__call__()`
        """

        return mars_cleaning_1st_pass(
            self.subarray.tel[tel_id].camera.geometry,
            image,
            picture_thresh=self.picture_threshold_pe.tel[tel_id],
            boundary_thresh=self.boundary_threshold_pe.tel[tel_id],
            min_number_picture_neighbors=self.min_picture_neighbors.tel[tel_id],
            keep_isolated_pixels=False,
        )


class FACTImageCleaner(TailcutsImageCleaner):
    """
    Clean images using the FACT technique. See `ctapipe.image.fact_image_cleaning`
    for algorithm details
    """

    time_limit_ns = FloatTelescopeParameter(
        default_value=5.0, help="arrival time limit for neighboring " "pixels, in ns"
    ).tag(config=True)

    def __call__(
        self, tel_id: int, image: np.ndarray, arrival_times=None
    ) -> np.ndarray:
        """ Apply FACT-style image cleaning. see ImageCleaner.__call__()"""
        return fact_image_cleaning(
            geom=self.subarray.tel[tel_id].camera.geometry,
            image=image,
            arrival_times=arrival_times,
            picture_threshold=self.picture_threshold_pe.tel[tel_id],
            boundary_threshold=self.boundary_threshold_pe.tel[tel_id],
            min_number_neighbors=self.min_picture_neighbors.tel[tel_id],
            time_limit=self.time_limit_ns.tel[tel_id],
        )
