"""
Image Cleaning Algorithms (identification of noisy pixels)


All algorithms return a boolean mask that is True for pixels surviving
the cleaning.

To get a zero-suppressed image and pixel
list, use ``image[mask], geom.pix_id[mask]``, or to keep the same
image size and just set unclean pixels to 0 or similar, use
``image[~mask] = 0``

"""

__all__ = [
    "tailcuts_clean",
    "bright_cleaning",
    "dilate",
    "mars_cleaning_1st_pass",
    "fact_image_cleaning",
    "apply_time_delta_cleaning",
    "apply_time_average_cleaning",
    "time_constrained_clean",
    "nsb_image_cleaning",
    "ImageCleaner",
    "TailcutsImageCleaner",
    "NSBImageCleaner",
    "MARSImageCleaner",
    "FACTImageCleaner",
    "TimeConstrainedImageCleaner",
]

from abc import abstractmethod

import numpy as np

from ctapipe.image.statistics import n_largest

from ..containers import MonitoringCameraContainer
from ..core import TelescopeComponent
from ..core.traits import (
    BoolTelescopeParameter,
    FloatTelescopeParameter,
    IntTelescopeParameter,
)
from .morphology import brightest_island, largest_island, number_of_islands


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
    geom : `ctapipe.instrument.CameraGeometry`
        Camera geometry information
    image : np.ndarray
        pixel charges
    picture_thresh : float | np.ndarray
        threshold above which all pixels are retained
    boundary_thresh : float | np.ndarray
        threshold above which pixels are retained if they have a neighbor
        already above the picture_thresh
    keep_isolated_pixels : bool
        If True, pixels above the picture threshold will be included always,
        if not they are only included if a neighbor is in the picture or
        boundary
    min_number_picture_neighbors : int
        A picture pixel survives cleaning only if it has at least this number
        of picture neighbors. This has no effect in case keep_isolated_pixels is True

    Returns
    -------
    A boolean mask of selected pixels.
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


def bright_cleaning(image, threshold, fraction, n_pixels=3):
    """
    Clean an image by removing pixels below a fraction of the mean charge
    in the ``n_pixels`` brightest pixels.

    No pixels are removed instead if the mean charge of the brightest pixels
    are below a certain threshold.

    Parameters
    ----------
    image : np.ndarray
        pixel charges
    threshold : float
        Minimum average charge in the ``n_pixels`` brightest pixels to apply
        cleaning
    fraction : float
        Pixels below fraction * (average charge in the ``n_pixels`` brightest pixels)
        will be removed from the cleaned image
    n_pixels : int
        Consider this number of the brightest pixels to calculate the mean charge

    Returns
    -------
    A boolean mask of selected pixels.

    """
    mean_brightest_signal = np.mean(n_largest(n_pixels, image))

    if mean_brightest_signal < threshold:
        return np.ones(image.shape, dtype=bool)

    threshold_brightest = fraction * mean_brightest_signal
    mask = image >= threshold_brightest

    return mask


def mars_cleaning_1st_pass(
    geom,
    image,
    picture_thresh=7,
    boundary_thresh=5,
    keep_isolated_pixels=False,
    min_number_picture_neighbors=0,
):
    """
    Clean an image by selecting pixels that pass a three-threshold tail-cuts
    procedure.

    All thresholds are defined with respect to the pedestal
    dispersion. All pixels that have a signal higher than the core (picture)
    threshold will be retained, along with all those above the boundary
    threshold that are neighbors of a core pixel AND all those above
    the boundary threshold that are neighbors of a neighbor of a core pixel.

    Parameters
    ----------
    geom : `ctapipe.instrument.CameraGeometry`
        Camera geometry information
    image : np.ndarray
        pixel charges
    picture_thresh : float
        threshold above which all pixels are retained
    boundary_thresh : float
        threshold above which pixels are retained if
        they have a neighbor already above the ``picture_thresh``; it is then
        reapplied iteratively to the neighbor of the neighbor
    keep_isolated_pixels : bool
        If True, pixels above the picture threshold will be included always,
        if not they are only included if a neighbor is in the picture or
        boundary
    min_number_picture_neighbors : int
        A picture pixel survives cleaning only if it has at least this number
        of picture neighbors. This has no effect in case ``keep_isolated_pixels`` is True

    Returns
    -------
    A boolean mask of selected pixels.
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
    Add one row of neighbors to the true values of a pixel mask and return
    the new mask.

    This can be used to include extra rows of pixels in a mask that was
    pre-computed, e.g. via `tailcuts_clean`.

    Parameters
    ----------
    geom : `ctapipe.instrument.CameraGeometry`
        Camera geometry information
    mask : np.ndarray
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
    geom : `ctapipe.instrument.CameraGeometry`
        Camera geometry information
    mask : np.ndarray
        boolean mask of selected pixels before `apply_time_delta_cleaning`
    arrival_times : np.ndarray
        pixel timing information
    min_number_neighbors : int
        a selected pixel needs at least this number of (already selected) neighbors
        that arrived within a given time_limit to itself to survive the cleaning.
    time_limit : int | float
        arrival time limit for neighboring pixels

    Returns
    -------
    A boolean mask of selected pixels.
    """
    pixels_to_keep = mask.copy()
    time_diffs = np.abs(arrival_times[mask, None] - arrival_times)
    # neighboring pixels arriving in the time limit and previously selected
    valid_neighbors = (time_diffs < time_limit) & geom.neighbor_matrix[mask] & mask
    enough_neighbors = np.count_nonzero(valid_neighbors, axis=1) >= min_number_neighbors
    pixels_to_keep[pixels_to_keep] &= enough_neighbors
    return pixels_to_keep


def apply_time_average_cleaning(
    geom, mask, image, arrival_times, picture_thresh, time_limit
):
    """
    Extract all pixels that arrived within a given timeframe
    with respect to the time average of the pixels on the main island.

    In order to avoid removing signal pixels of large impact-parameter events,
    the time limit for bright pixels is doubled.

    Parameters
    ----------
    geom : `ctapipe.instrument.CameraGeometry`
        Camera geometry information
    mask : np.ndarray
        boolean mask of selected pixels before `apply_time_delta_cleaning`
    image : np.ndarray
        pixel charges
    arrival_times : np.ndarray
        pixel timing information
    picture_thresh : float
        threshold above which ``time_limit`` is extended twice its value
    time_limit : int | float
        arrival time limit w.r.t. the average time of the core pixels

    Returns
    -------
    A boolean mask of selected pixels.
    """
    mask = mask.copy()
    if np.count_nonzero(mask) > 0:
        # use main island (maximum charge) for time average calculation
        n_islands, island_labels = number_of_islands(geom, mask)
        mask_main = brightest_island(n_islands, island_labels, image)
        time_ave = np.average(arrival_times[mask_main], weights=image[mask_main] ** 2)

        time_diffs = np.abs(arrival_times[mask] - time_ave)
        time_limit_pixwise = np.where(
            image < (2 * picture_thresh), time_limit, time_limit * 2
        )[mask]

        mask[mask] &= time_diffs < time_limit_pixwise

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
    geom : `ctapipe.instrument.CameraGeometry`
        Camera geometry information
    image : np.ndarray
        pixel charges
    arrival_times : np.ndarray
        pixel timing information
    picture_threshold : float | np.ndarray
        threshold above which all pixels are retained
    boundary_threshold : float | np.ndarray
        threshold above which pixels are retained if they have a neighbor
        already above the ``picture_thresh``
    min_number_neighbors : int
        Threshold to determine if a pixel survives cleaning steps.
        These steps include checks of neighbor arrival time and value
    time_limit : int | float
        arrival time limit for neighboring pixels

    Returns
    -------
    A boolean mask of selected pixels.

    References
    ----------
    See :cite:p:`phd-temme` and for implementation :cite:p:`fact-tools`.
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


def time_constrained_clean(
    geom,
    image,
    arrival_times,
    picture_thresh=7,
    boundary_thresh=5,
    time_limit_core=4.5,
    time_limit_boundary=1.5,
    min_number_picture_neighbors=1,
):
    """
    Time constrained cleaning by MAGIC

    Cleaning contains the following steps:
    - Find core pixels (containing more photons than a picture threshold)
    - Remove pixels with less than N neighbors
    - Keep core pixels whose arrival times are within a time limit of the average time
    - Find boundary pixels (containing more photons than a boundary threshold)
    - Remove pixels with less than N neighbors arriving within a given timeframe

    Parameters
    ----------
    geom : `ctapipe.instrument.CameraGeometry`
        Camera geometry information
    image : np.ndarray
        pixel charges
    arrival_times : np.ndarray
        pixel timing information
    picture_threshold : float | np.ndarray
        threshold above which all pixels are retained
    boundary_threshold : float | np.ndarray
        threshold above which pixels are retained if they have a neighbor
        already above the ``picture_thresh``
    time_limit_core : int | float
        arrival time limit of core pixels w.r.t the average time
    time_limit_boundary : int | float
        arrival time limit of boundary pixels w.r.t neighboring core pixels
    min_number_neighbors : int
        Threshold to determine if a pixel survives cleaning steps.
        These steps include checks of neighbor arrival time and value

    Returns
    -------
    A boolean mask of selected pixels.
    """

    # find core pixels that pass a picture threshold
    pixels_above_picture = image >= picture_thresh

    # require at least min_number_picture_neighbors
    number_of_neighbors_above_picture = geom.neighbor_matrix_sparse.dot(
        pixels_above_picture.view(np.byte)
    )
    pixels_in_picture = pixels_above_picture & (
        number_of_neighbors_above_picture >= min_number_picture_neighbors
    )

    # keep core pixels whose arrival times are within a certain time limit of the average
    mask_core = apply_time_average_cleaning(
        geom, pixels_in_picture, image, arrival_times, picture_thresh, time_limit_core
    )

    # find boundary pixels that pass a boundary threshold
    pixels_above_boundary = image >= boundary_thresh
    pixels_with_picture_neighbors = geom.neighbor_matrix_sparse.dot(mask_core)
    mask_boundary = (pixels_above_boundary & pixels_with_picture_neighbors) & np.invert(
        mask_core
    )

    # keep boundary pixels whose arrival times are within a certain time limit of the neighboring core pixels
    mask_boundary = mask_boundary.copy()

    time_diffs = np.abs(arrival_times[mask_boundary, None] - arrival_times)
    valid_neighbors = (
        (time_diffs < time_limit_boundary)
        & geom.neighbor_matrix[mask_boundary]
        & mask_core
    )
    enough_neighbors = (
        np.count_nonzero(valid_neighbors, axis=1) >= min_number_picture_neighbors
    )
    mask_boundary[mask_boundary] &= enough_neighbors

    return mask_core | mask_boundary


def nsb_image_cleaning(
    geom,
    image,
    arrival_times,
    picture_thresh_min=8,
    boundary_thresh=4,
    min_number_picture_neighbors=2,
    keep_isolated_pixels=False,
    time_limit=None,
    time_num_neighbors=1,
    bright_cleaning_n_pixels=3,
    bright_cleaning_fraction=0.03,
    bright_cleaning_threshold=None,
    largest_island_only=False,
    pedestal_factor=2.5,
    pedestal_std=None,
):
    """
    Clean an image in 5 Steps:

    1) Get pixelwise picture thresholds for `tailcuts_clean` in step 2) from interleaved
       pedestal events if ``pedestal_std`` is not None.
    2) Apply `tailcuts_clean` algorithm.
    3) Apply `apply_time_delta_cleaning` algorithm if ``time_limit`` is not None.
    4) Apply `bright_cleaning` if ``bright_cleaning_threshold`` is not None.
    5) Get only `ctapipe.image.largest_island` if ``largest_island_only`` is
       set to true.

    Parameters
    ----------
    geom : `ctapipe.instrument.CameraGeometry`
        Camera geometry information
    image : np.ndarray
        Pixel charges
    arrival_times : np.ndarray
        Pixel timing information
    picture_thresh_min : float | np.ndarray
        Defines the minimum value used for the picture threshold for `tailcuts_clean`.
        The threshold used will be at least this value, or higher if ``pedestal_std``
        and ``pedestal_factor`` are set.
    boundary_thresh : float | np.ndarray
        Threshold above which pixels are retained if they have a neighbor
        already above the ``picture_thresh_min``. Used for `tailcuts_clean`.
    min_number_picture_neighbors : int
        A picture pixel survives cleaning only if it has at least this number
        of picture neighbors. This has no effect in case ``keep_isolated_pixels`` is True.
        Used for `tailcuts_clean`.
    keep_isolated_pixels : bool
        If True, pixels above the picture threshold will be included always,
        if not they are only included if a neighbor is in the picture or
        boundary. Used for `tailcuts_clean`.
    time_limit : float
        Time limit for the `apply_time_delta_cleaning`. Set to None if no
        `apply_time_delta_cleaning` should be applied.
    time_num_neighbors : int
        Used for `apply_time_delta_cleaning`.
        A selected pixel needs at least this number of (already selected) neighbors
        that arrived within a given ``time_limit`` to itself to survive this cleaning.
    bright_cleaning_n_pixels : int
        Consider this number of the brightest pixels for calculating the mean charge.
        Pixels below fraction * (mean charge in the ``bright_cleaning_n_pixels``
        brightest pixels) will be removed from the cleaned image. Set
        ``bright_cleaning_threshold`` to None if no `bright_cleaning` should be applied.
    bright_cleaning_fraction : float
        Fraction parameter for `bright_cleaning`. Pixels below
        fraction * (mean charge in the ``bright_cleaning_n_pixels`` brightest pixels)
        will be removed from the cleaned image. Set ``bright_cleaning_threshold`` to None
        if no `bright_cleaning` should be applied.
    bright_cleaning_threshold : float
        Threshold parameter for `bright_cleaning`. Minimum mean charge
        in the ``bright_cleaning_n_pixels`` brightest pixels to apply the cleaning.
        Set to None if no `bright_cleaning` should be applied.
    largest_island_only : bool
        Set to true to get only largest island.
    pedestal_factor : float
        Factor for interleaved pedestal cleaning. It is multiplied by the
        pedestal standard deviation for each pixel to calculate pixelwise picture
        threshold parameters for `tailcuts_clean` considering the current background.
        Has no effect if ``pedestal_std`` is set to None.
    pedestal_std : np.ndarray | None
        Pedestal standard deviation for each pixel. See
        `ctapipe.containers.PedestalContainer`. Used to calculate pixelwise picture
        threshold parameters for `tailcuts_clean` by multiplying it with ``pedestal_factor``
        and clip (limit) the product with ``picture_thresh_min``. If set to None, only
        ``picture_thresh_min`` is used to set the picture threshold for `tailcuts_clean`.

    Returns
    -------
    A boolean mask of selected pixels.

    """
    # Step 1
    picture_thresh = picture_thresh_min
    if pedestal_std is not None:
        pedestal_threshold = pedestal_std * pedestal_factor
        picture_thresh = np.clip(pedestal_threshold, picture_thresh_min, None)

    # Step 2
    mask = tailcuts_clean(
        geom,
        image,
        picture_thresh=picture_thresh,
        boundary_thresh=boundary_thresh,
        min_number_picture_neighbors=min_number_picture_neighbors,
        keep_isolated_pixels=keep_isolated_pixels,
    )
    # Check that at least one pixel survives tailcuts_clean
    if np.count_nonzero(mask) == 0:
        return mask

    # Step 3
    if time_limit is not None:
        mask = apply_time_delta_cleaning(
            geom,
            mask,
            arrival_times,
            min_number_neighbors=time_num_neighbors,
            time_limit=time_limit,
        )

    # Step 4
    if bright_cleaning_threshold is not None:
        mask &= bright_cleaning(
            image,
            bright_cleaning_threshold,
            bright_cleaning_fraction,
            bright_cleaning_n_pixels,
        )

    # Step 5
    if largest_island_only:
        _, island_labels = number_of_islands(geom, mask)
        mask = largest_island(island_labels)

    return mask


class ImageCleaner(TelescopeComponent):
    """
    Abstract class for all configurable Image Cleaning algorithms. Use
    ``ImageCleaner.from_name()`` to construct an instance of a particular algorithm
    """

    @abstractmethod
    def __call__(
        self,
        tel_id: int,
        image: np.ndarray,
        arrival_times: np.ndarray = None,
        *,
        monitoring: MonitoringCameraContainer = None,
    ) -> np.ndarray:
        """
        Identify pixels with signal, and reject those with pure noise.

        Parameters
        ----------
        tel_id : int
            which telescope id in the subarray is being used (determines
            which cut is used)
        image : np.ndarray
            image pixel data corresponding to the camera geometry
        arrival_times : np.ndarray
            image of arrival time (not used in this method)
        monitoring : `ctapipe.containers.MonitoringCameraContainer`
            `ctapipe.containers.MonitoringCameraContainer` to make use of
            additional parameters from monitoring data e.g. pedestal std.

        Returns
        -------
        np.ndarray
            boolean mask of pixels passing cleaning
        """
        pass


class TailcutsImageCleaner(ImageCleaner):
    """
    Clean images using the standard picture/boundary technique. See
    `ctapipe.image.tailcuts_clean`.
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
        self,
        tel_id: int,
        image: np.ndarray,
        arrival_times: np.ndarray = None,
        *,
        monitoring: MonitoringCameraContainer = None,
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


class NSBImageCleaner(TailcutsImageCleaner):
    """
    Clean images based on lstchains image cleaning technique described in
    :cite:p:`lst1-crab-paper`. See `ctapipe.image.nsb_image_cleaning`.
    """

    time_limit = FloatTelescopeParameter(
        default_value=2,
        help="Time limit for the `apply_time_delta_cleaning`. Set to None if no"
        " `apply_time_delta_cleaning` should be applied.",
        allow_none=True,
    ).tag(config=True)

    time_num_neighbors = IntTelescopeParameter(
        default_value=1,
        help="Used for `apply_time_delta_cleaning`."
        " A selected pixel needs at least this number of (already selected) neighbors"
        " that arrived within a given `time_limit` to itself to survive this cleaning.",
    ).tag(config=True)

    bright_cleaning_n_pixels = IntTelescopeParameter(
        default_value=3,
        help="Consider this number of the brightest pixels for calculating the"
        " mean charge. Pixels below fraction * (mean charge in the"
        " ``bright_cleaning_n_pixels`` brightest pixels) will be removed from the"
        " cleaned image. Set ``bright_cleaning_threshold`` to None if no"
        " `bright_cleaning` should be applied.",
    ).tag(config=True)

    bright_cleaning_fraction = FloatTelescopeParameter(
        default_value=0.03,
        help="Fraction parameter for `bright_cleaning`. Pixels below"
        " fraction * (mean charge in the ``bright_cleaning_n_pixels`` brightest pixels)"
        " will be removed from the cleaned image. Set ``bright_cleaning_threshold`` to"
        " None if no `bright_cleaning` should be applied.",
    ).tag(config=True)

    bright_cleaning_threshold = FloatTelescopeParameter(
        default_value=267,
        help="Threshold parameter for `bright_cleaning`. Minimum mean charge"
        " in the ``bright_cleaning_n_pixels`` brightest pixels to apply the cleaning."
        " Set to None if no `bright_cleaning` should be applied.",
        allow_none=True,
    ).tag(config=True)

    largest_island_only = BoolTelescopeParameter(
        default_value=False, help="Set to true to get only largest island"
    ).tag(config=True)

    pedestal_factor = FloatTelescopeParameter(
        default_value=2.5,
        help="Factor for interleaved pedestal cleaning. It is multiplied by the"
        " pedestal standard deviation for each pixel to calculate pixelwise upper limit"
        " picture thresholds and clip them with ``picture_thresh_pe`` of"
        " `TailcutsImageCleaner` for `tailcuts_clean` considering the current background."
        " If no pedestal standard deviation is given, interleaved pedestal cleaning is"
        " not applied and ``picture_thresh_pe`` of `TailcutsImageCleaner` is used alone"
        " instead.",
    ).tag(config=True)

    def __call__(
        self,
        tel_id: int,
        image: np.ndarray,
        arrival_times: np.ndarray = None,
        *,
        monitoring: MonitoringCameraContainer = None,
    ) -> np.ndarray:
        """
        Apply NSB image cleaning used by lstchain. See `ImageCleaner.__call__()`
        """
        pedestal_std = None
        if monitoring is not None:
            pedestal_std = monitoring.pedestal.charge_std

        return nsb_image_cleaning(
            self.subarray.tel[tel_id].camera.geometry,
            image,
            arrival_times,
            picture_thresh_min=self.picture_threshold_pe.tel[tel_id],
            boundary_thresh=self.boundary_threshold_pe.tel[tel_id],
            min_number_picture_neighbors=self.min_picture_neighbors.tel[tel_id],
            keep_isolated_pixels=self.keep_isolated_pixels.tel[tel_id],
            time_limit=self.time_limit.tel[tel_id],
            time_num_neighbors=self.time_num_neighbors.tel[tel_id],
            bright_cleaning_n_pixels=self.bright_cleaning_n_pixels.tel[tel_id],
            bright_cleaning_fraction=self.bright_cleaning_fraction.tel[tel_id],
            bright_cleaning_threshold=self.bright_cleaning_threshold.tel[tel_id],
            largest_island_only=self.largest_island_only.tel[tel_id],
            pedestal_factor=self.pedestal_factor.tel[tel_id],
            pedestal_std=pedestal_std,
        )


class MARSImageCleaner(TailcutsImageCleaner):
    """
    1st-pass MARS-like Image cleaner (See `ctapipe.image.mars_cleaning_1st_pass`)
    """

    def __call__(
        self,
        tel_id: int,
        image: np.ndarray,
        arrival_times: np.ndarray = None,
        *,
        monitoring: MonitoringCameraContainer = None,
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
    for algorithm details.
    """

    time_limit_ns = FloatTelescopeParameter(
        default_value=5.0, help="arrival time limit for neighboring pixels, in ns"
    ).tag(config=True)

    def __call__(
        self,
        tel_id: int,
        image: np.ndarray,
        arrival_times: np.ndarray = None,
        *,
        monitoring: MonitoringCameraContainer = None,
    ) -> np.ndarray:
        """Apply FACT-style image cleaning. see ImageCleaner.__call__()"""

        return fact_image_cleaning(
            geom=self.subarray.tel[tel_id].camera.geometry,
            image=image,
            arrival_times=arrival_times,
            picture_threshold=self.picture_threshold_pe.tel[tel_id],
            boundary_threshold=self.boundary_threshold_pe.tel[tel_id],
            min_number_neighbors=self.min_picture_neighbors.tel[tel_id],
            time_limit=self.time_limit_ns.tel[tel_id],
        )


class TimeConstrainedImageCleaner(TailcutsImageCleaner):
    """
    MAGIC-like Image cleaner with timing information (See
    `ctapipe.image.time_constrained_clean`).
    """

    time_limit_core_ns = FloatTelescopeParameter(
        default_value=4.5,
        help="arrival time limit for neighboring core pixels, in ns",
    ).tag(config=True)
    time_limit_boundary_ns = FloatTelescopeParameter(
        default_value=1.5,
        help="arrival time limit for neighboring boundary pixels, in ns",
    ).tag(config=True)

    def __call__(
        self,
        tel_id: int,
        image: np.ndarray,
        arrival_times: np.ndarray = None,
        *,
        monitoring: MonitoringCameraContainer = None,
    ) -> np.ndarray:
        """
        Apply MAGIC-like image cleaning with timing information. See `ImageCleaner.__call__()`
        """

        return time_constrained_clean(
            self.subarray.tel[tel_id].camera.geometry,
            image,
            arrival_times=arrival_times,
            picture_thresh=self.picture_threshold_pe.tel[tel_id],
            boundary_thresh=self.boundary_threshold_pe.tel[tel_id],
            min_number_picture_neighbors=self.min_picture_neighbors.tel[tel_id],
            time_limit_core=self.time_limit_core_ns.tel[tel_id],
            time_limit_boundary=self.time_limit_boundary_ns.tel[tel_id],
        )
