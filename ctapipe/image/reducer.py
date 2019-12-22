"""
Algorithms for the data volume reduction.
"""
from abc import abstractmethod
import numpy as np
from ctapipe.core import Component, traits
from ctapipe.image.extractor import LocalPeakWindowSum
from ctapipe.image.cleaning import (
    tailcuts_clean,
    dilate
)

__all__ = [
    'DataVolumeReducer',
    'NullDataVolumeReducer',
    'TailCutsDataVolumeReducer',
]


class DataVolumeReducer(Component):
    """
    Base component for data volume reducers.
    """

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
        mask : array
            Mask of selected pixels.
        """
        mask = self.select_pixels(waveforms)
        return mask

    @abstractmethod
    def select_pixels(self, waveforms):
        """
        Abstract method to be defined by a DataVolumeReducer subclass.

        Call the relevant functions for the required pixel selection.

        Parameters
        ----------
        waveforms : ndarray
            Waveforms stored in a numpy array of shape
            (n_pix, n_samples).

        Returns
        -------
        mask : array
            Mask of selected pixels.
        """


class NullDataVolumeReducer(DataVolumeReducer):
    """
    Perform no data volume reduction
    """

    def select_pixels(self, waveforms):
        mask = waveforms != 0
        return mask


class TailCutsDataVolumeReducer(DataVolumeReducer):
    """
    Reduce the time integrated shower image in 3 Steps:

    1) Select pixels with tailcuts_clean.
    2) Add iteratively all pixels with Signal S >= boundary_thresh
       with ctapipe module dilate until no new pixels were added.
    3) Adding new pixels with dilate to get more conservative.
    """
    camera_geom = traits.Any(
        help="Get camera geometry information from "
             "ctapipe.instrument.CameraGeometry."
    ).tag(config=True)
    end_dilates = traits.Integer(
        default_value=1,
        help="Number of how many times to dilate at the end."
    ).tag(config=True)
    picture_thresh = traits.Float(
        default_value=10,
        help="Picture threshold for the first tailcuts_clean step. All "
             "pixels above are selected."
    ).tag(config=True)
    boundary_thresh = traits.Float(
        default_value=5,
        help="Boundary threshold for the first tailcuts_clean step and "
             "also for the second iteration step."
    ).tag(config=True)
    keep_isolated_pixels = traits.Bool(
        default_value=True,
        help="If True, pixels above the picture threshold will be included "
             "always, if not they are only included if a neighbor is in "
             "the picture or boundary."
    ).tag(config=True)
    min_number_picture_neighbors = traits.Integer(
        default_value=0,
        help="A picture pixel survives tailcuts_clean only if it has at "
             "least this number of picture neighbors. This has no effect "
             "in case keep_isolated_pixels is True."
    ).tag(config=True)

    def select_pixels(self, waveforms):
        """
        Parameters
        ----------
        waveforms : ndarray
                Waveforms stored in a numpy array of shape
                (n_pix, n_samples).

        Traitlets:
        camera_geom: 'ctapipe.instrument.CameraGeometry'
            Camera geometry information
        picture_thresh: float
            threshold for tailcuts_clean. All pixels above are retained
        boundary_thresh: float
            1)Threshold for tailcuts_clean. All pixels above are retained if
            they have a neighbor already above the picture_thresh.
            2)Threshold for the iteration step 2). All pixels above are
            selected.
        keep_isolated_pixels: bool
            For tailcuts_clean: If True, pixels above the picture threshold
            will be included always, if not they are only included if a
            neighbor is in the picture or boundary.
        min_number_picture_neighbors: int
            For tailcuts_clean: A picture pixel survives cleaning only if it
            has at least this number of picture neighbors. This has no effect
            in case keep_isolated_pixels is True
        end_dilates: int
            Number of how many times to dilate at the end in Step 3).

        Returns
        -------
        mask : array
            Mask of selected pixels.
        """
        # Pulse-integrate waveforms
        image_extractor = LocalPeakWindowSum()
        charge, _ = image_extractor(waveforms)

        # 1) Step: TailcutCleaning at first
        mask = tailcuts_clean(
            geom=self.camera_geom,
            image=charge,
            picture_thresh=self.picture_thresh,
            boundary_thresh=self.boundary_thresh,
            keep_isolated_pixels=self.keep_isolated_pixels,
            min_number_picture_neighbors=self.min_number_picture_neighbors
        )
        pixels_above_boundary_thresh = charge >= self.boundary_thresh
        mask_in_loop = np.array([])
        # 2) Step: Add iteratively all pixels with Signal
        #          S > boundary_thresh with ctapipe module
        #          'dilate' until no new pixels were added.
        while not np.array_equal(mask, mask_in_loop):
            mask_in_loop = mask
            mask = dilate(self.camera_geom, mask) & pixels_above_boundary_thresh

        # 3) Step: Adding Pixels with 'dilate' to get more conservative.
        for _ in range(self.end_dilates):
            mask = dilate(self.camera_geom, mask)

        return mask
