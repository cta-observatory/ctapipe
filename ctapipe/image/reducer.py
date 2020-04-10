"""
Algorithms for the data volume reduction.
"""
from abc import abstractmethod
import numpy as np
from ctapipe.core import Component, traits
from ctapipe.image.extractor import NeighborPeakWindowSum
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

    def __init__(
        self,
        config=None,
        parent=None,
        subarray=None,
        image_extractor=None,
        **kwargs
    ):
        """
        Parameters
        ----------
        config: traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool: ctapipe.core.Tool or None
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        subarray: ctapipe.instrument.SubarrayDescription
            Description of the subarray
        image_extractor: ctapipe.image.extractor.ImageExtractor
            The ImageExtractor to use for 'TailCutsDataVolumeReducer'.
            If None, then NeighborPeakWindowSum will be used by default.
        kwargs
        """

        super().__init__(config=config, parent=parent, **kwargs)
        self.subarray = subarray

        if image_extractor is None:
            image_extractor = NeighborPeakWindowSum(parent=self, subarray=subarray)
        self.image_extractor = image_extractor

    def __call__(self, waveforms, telid=None, selected_gain_channel=None):
        """
        Call the relevant functions to perform data volume reduction on the
        waveforms.

        Parameters
        ----------
        waveforms: ndarray
            Waveforms stored in a numpy array of shape
            (n_pix, n_samples).
        telid: int
            The telescope id. Required for the 'image_extractor' and
            'camera.geometry' in 'TailCutsDataVolumeReducer'.
        selected_gain_channel: ndarray
            The channel selected in the gain selection, per pixel. Required for
            the 'image_extractor' in 'TailCutsDataVolumeReducer'.
            extraction.

        Returns
        -------
        mask: array
            Mask of selected pixels.
        """

        mask = self.select_pixels(
            waveforms, telid=telid, selected_gain_channel=selected_gain_channel
        )
        return mask

    @abstractmethod
    def select_pixels(self, waveforms, telid=None, selected_gain_channel=None):
        """
        Abstract method to be defined by a DataVolumeReducer subclass.
        Call the relevant functions for the required pixel selection.

        Parameters
        ----------
        waveforms: ndarray
            Waveforms stored in a numpy array of shape
            (n_pix, n_samples).
        telid: int
            The telescope id. Required for the 'image_extractor' and
            'camera.geometry' in 'TailCutsDataVolumeReducer'.
        selected_gain_channel: ndarray
            The channel selected in the gain selection, per pixel. Required for
            the 'image_extractor' in 'TailCutsDataVolumeReducer'.

        Returns
        -------
        mask: array
            Mask of selected pixels.
        """


class NullDataVolumeReducer(DataVolumeReducer):
    """
    Perform no data volume reduction
    """

    def select_pixels(self, waveforms, telid=None, selected_gain_channel=None):
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

    def select_pixels(self, waveforms, telid=None, selected_gain_channel=None):
        camera_geom = self.subarray.tel[telid].camera.geometry
        # Pulse-integrate waveforms
        charge, _ = self.image_extractor(
            waveforms, telid=telid, selected_gain_channel=selected_gain_channel
        )

        # 1) Step: TailcutCleaning at first
        mask = tailcuts_clean(
            geom=camera_geom,
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
            mask = dilate(camera_geom, mask) & pixels_above_boundary_thresh

        # 3) Step: Adding Pixels with 'dilate' to get more conservative.
        for _ in range(self.end_dilates):
            mask = dilate(camera_geom, mask)

        return mask
