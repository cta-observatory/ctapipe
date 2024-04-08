"""
Algorithms for the data volume reduction.
"""

from abc import abstractmethod

import numpy as np

from ctapipe.containers import DL1CameraContainer
from ctapipe.core import TelescopeComponent
from ctapipe.core.traits import (
    BoolTelescopeParameter,
    ComponentName,
    IntTelescopeParameter,
    TelescopeParameter,
)
from ctapipe.image import TailcutsImageCleaner
from ctapipe.image.cleaning import dilate
from ctapipe.image.extractor import ImageExtractor

__all__ = ["DataVolumeReducer", "NullDataVolumeReducer", "TailCutsDataVolumeReducer"]


class DataVolumeReducer(TelescopeComponent):
    """
    Base component for data volume reducers.
    """

    def __init__(self, subarray, config=None, parent=None, **kwargs):
        """
        Parameters
        ----------
        subarray: ctapipe.instrument.SubarrayDescription
            Description of the subarray
        config: traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        kwargs
        """
        self.subarray = subarray
        super().__init__(config=config, parent=parent, subarray=subarray, **kwargs)

    def __call__(self, waveforms, tel_id=None, selected_gain_channel=None):
        """
        Call the relevant functions to perform data volume reduction on the
        waveforms.

        Parameters
        ----------
        waveforms: ndarray
            Waveforms stored in a numpy array of shape
            (n_pix, n_samples).
        tel_id: int
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
            waveforms, tel_id=tel_id, selected_gain_channel=selected_gain_channel
        )
        return mask

    @abstractmethod
    def select_pixels(self, waveforms, tel_id=None, selected_gain_channel=None):
        """
        Abstract method to be defined by a DataVolumeReducer subclass.
        Call the relevant functions for the required pixel selection.

        Parameters
        ----------
        waveforms: ndarray
            Waveforms stored in a numpy array of shape
            (n_pix, n_samples).
        tel_id: int
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

    def select_pixels(self, waveforms, tel_id=None, selected_gain_channel=None):
        n_pixels = waveforms.shape[-2]
        return np.ones(n_pixels, dtype=bool)


class TailCutsDataVolumeReducer(DataVolumeReducer):
    """
    Reduce the time integrated shower image in 3 Steps:

    1) Select pixels with tailcuts_clean.
    2) Add iteratively all pixels with Signal S >= boundary_thresh
       with ctapipe module dilate until no new pixels were added.
    3) Adding new pixels with dilate to get more conservative.

    Attributes
    ----------
    image_extractor_type: String
        Name of the image_extractor to be used.
    n_end_dilates: IntTelescopeParameter
        Number of how many times to dilate at the end.
    do_boundary_dilation: BoolTelescopeParameter
        If set to 'False', the iteration steps in 2) are skipped and
        normal TailcutCleaning is used.
    """

    image_extractor_type = TelescopeParameter(
        trait=ComponentName(ImageExtractor, default_value="NeighborPeakWindowSum"),
        default_value="NeighborPeakWindowSum",
        help="Name of the ImageExtractor subclass to be used.",
    ).tag(config=True)

    n_end_dilates = IntTelescopeParameter(
        default_value=1, help="Number of how many times to dilate at the end."
    ).tag(config=True)

    do_boundary_dilation = BoolTelescopeParameter(
        default_value=True,
        help="If set to 'False', the iteration steps in 2) are skipped and"
        "normal TailcutCleaning is used.",
    ).tag(config=True)

    def __init__(
        self,
        subarray,
        config=None,
        parent=None,
        cleaner=None,
        image_extractor=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        subarray: ctapipe.instrument.SubarrayDescription
            Description of the subarray
        config: traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        kwargs
        """
        super().__init__(config=config, parent=parent, subarray=subarray, **kwargs)

        if cleaner is None:
            self.cleaner = TailcutsImageCleaner(parent=self, subarray=self.subarray)
        else:
            self.cleaner = cleaner

        self.image_extractors = {}
        if image_extractor is None:
            for _, _, name in self.image_extractor_type:
                self.image_extractors[name] = ImageExtractor.from_name(
                    name, subarray=self.subarray, parent=self
                )
        else:
            name = image_extractor.__class__.__name__
            self.image_extractor_type = [("type", "*", name)]
            self.image_extractors[name] = image_extractor

    def select_pixels(self, waveforms, tel_id=None, selected_gain_channel=None):
        camera_geom = self.subarray.tel[tel_id].camera.geometry
        # Pulse-integrate waveforms
        extractor = self.image_extractors[self.image_extractor_type.tel[tel_id]]
        # do not treat broken pixels in data volume reduction
        broken_pixels = np.zeros(
            (waveforms.shape[-3], camera_geom.n_pixels), dtype=bool
        )
        dl1: DL1CameraContainer = extractor(
            waveforms,
            tel_id=tel_id,
            selected_gain_channel=selected_gain_channel,
            broken_pixels=broken_pixels,
        )

        # 1) Step: TailcutCleaning at first
        mask = self.cleaner(tel_id, dl1.image)
        pixels_above_boundary_thresh = (
            dl1.image >= self.cleaner.boundary_threshold_pe.tel[tel_id]
        )
        mask_in_loop = np.array([])
        # 2) Step: Add iteratively all pixels with Signal
        #          S > boundary_thresh with ctapipe module
        #          'dilate' until no new pixels were added.
        while (
            not np.array_equal(mask, mask_in_loop)
            and self.do_boundary_dilation.tel[tel_id]
        ):
            mask_in_loop = mask
            mask = dilate(camera_geom, mask) & pixels_above_boundary_thresh

        # 3) Step: Adding Pixels with 'dilate' to get more conservative.
        for _ in range(self.n_end_dilates.tel[tel_id]):
            mask = dilate(camera_geom, mask)

        return mask
