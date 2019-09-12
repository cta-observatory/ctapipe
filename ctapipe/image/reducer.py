"""
Algorithms for the data volume reduction.
"""

from abc import abstractmethod

import numpy as np
from ctapipe.core import Component
from ctapipe.image.cleaning import tailcuts_clean, dilate

__all__ = [
    'DataVolumeReducer',
    'NullDataVolumeReducer',
    'TailCutsDataVolumeReducer',
]


class DataVolumeReducer(Component):
    """
    Base component for data volume reducers.
    """

    @abstractmethod
    def __call__(self, waveforms):
        """
        Call the relevant functions to perform data volume reduction on the
        waveforms.

        Parameters
        ----------
        waveforms : ndarray
            Waveforms stored in a numpy array of shape
            (n_pix, n_samples).

        Returns
        -------
        reduced_waveforms : ndarray
            Reduced waveforms stored in a numpy array of shape
            (n_pix, n_samples).
        """


class NullDataVolumeReducer(DataVolumeReducer):
    """
    Perform no data volume reduction
    """

    def __call__(self, waveforms):
        return waveforms


class TailCutsDataVolumeReducer(DataVolumeReducer):

    def __call__(
        self,
        geom,
        waveforms,
        end_dilates=1,
        picture_thresh=7,
        boundary_thresh=5,
        iteration_thresh=5,
        keep_isolated_pixels=True,
        min_number_picture_neighbors=0,
    ):
        """
        Reduce the image in 3 Steps:

        1) Select pixels with tailcuts_clean.
        2) Add iteratively all pixels with Signal S >= iteration_thresh
           with ctapipe module dilate until no new pixels were added.
        3) Adding new pixels with dilate to get more conservative.

        Parameters
        ----------
        geom: `ctapipe.instrument.CameraGeometry`
            Camera geometry information
        waveforms: ndarray
            Waveforms stored in a numpy array of shape
            (n_pix, n_samples).
        picture_thresh: float or array
            threshold for tailcuts_clean above which all pixels are retained
        boundary_thresh: float or array.
            threshold for tailcuts_clean above which pixels are retained if
            they have a neighbor already above the picture_thresh.
        keep_isolated_pixels: bool
            For tailcuts_clean: If True, pixels above the picture threshold
            will be included always, if not they are only included if a
            neighbor is in the picture or boundary.
        min_number_picture_neighbors: int
            For tailcuts_clean: A picture pixel survives cleaning only if it
            has at least this number of picture neighbors. This has no effect
            in case keep_isolated_pixels is True
        iteration_thresh: float
            Threshold for the iteration step 2), above which pixels are
            selected.
        end_dilates: int
            Number of how many times to dilate at the end in Step 3).

        Returns
        -------
        reduced_waveforms : ndarray
            Reduced waveforms stored in a numpy array of shape
            (n_pix, n_samples).

        """

        reduced_waveforms_mask = np.empty([waveforms.shape[0], 0], dtype=bool)

        for i in range(waveforms.shape[1]):
            image = waveforms[:, [i]]
            # 1) Step: TailcutCleaning at first
            mask = tailcuts_clean(
                geom=geom,
                image=image,
                picture_thresh=picture_thresh,
                boundary_thresh=boundary_thresh,
                keep_isolated_pixels=keep_isolated_pixels,
                min_number_picture_neighbors=min_number_picture_neighbors
            )
            pixels_above_iteration_thresh = image >= iteration_thresh
            mask_for_loop = np.array([])
            # 2) Step: Add iteratively all pixels with Signal
            #          S > iteration_thresh with ctapipe module
            #          'dilate' until no new pixels were added.
            while not np.array_equal(mask, mask_for_loop):
                mask_for_loop = mask
                mask = dilate(geom, mask) & pixels_above_iteration_thresh

            # 3) Step: Adding Pixels with 'dilate' to get more conservative.
            for p in range(end_dilates):
                mask = dilate(geom, mask)

            reduced_waveforms_mask = np.column_stack((reduced_waveforms_mask,
                                                      mask))

        return np.ma.masked_array(waveforms, mask=reduced_waveforms_mask)
