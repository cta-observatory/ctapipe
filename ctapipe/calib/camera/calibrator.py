"""
Definition of the `CameraCalibrator` class, providing all steps needed to apply
calibration and image extraction, as well as supporting algorithms.
"""

import warnings
import numpy as np
from ctapipe.core import Component
from ctapipe.image.extractor import NeighborPeakWindowSum
from ctapipe.image.reducer import NullDataVolumeReducer

__all__ = ["CameraCalibrator"]


class CameraCalibrator(Component):
    """
    Calibrator to handle the full camera calibration chain, in order to fill
    the DL1 data level in the event container.
    """

    def __init__(
        self,
        subarray,
        config=None,
        parent=None,
        data_volume_reducer=None,
        image_extractor=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        subarray: ctapipe.instrument.SubarrayDescription
            Description of the subarray. Provides information about the
            camera which are useful in calibration. Also required for
            configuring the TelescopeParameter traitlets.
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool : ctapipe.core.Tool or None
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        data_volume_reducer : ctapipe.image.reducer.DataVolumeReducer
            The DataVolumeReducer to use. If None, then
            NullDataVolumeReducer will be used by default, and waveforms
            will not be reduced.
        image_extractor : ctapipe.image.extractor.ImageExtractor
            The ImageExtractor to use. If None, then NeighborPeakWindowSum
            will be used by default.
        subarray: ctapipe.instrument.SubarrayDescription
            Description of the subarray
        kwargs
        """
        super().__init__(config=config, parent=parent, **kwargs)
        self.subarray = subarray

        self._r1_empty_warn = False
        self._dl0_empty_warn = False

        if image_extractor is None:
            image_extractor = NeighborPeakWindowSum(parent=self, subarray=subarray)
        self.image_extractor = image_extractor

        if data_volume_reducer is None:
            data_volume_reducer = NullDataVolumeReducer(parent=self, subarray=subarray)
        self.data_volume_reducer = data_volume_reducer

    def _check_r1_empty(self, waveforms):
        if waveforms is None:
            if not self._r1_empty_warn:
                warnings.warn(
                    "Encountered an event with no R1 data. "
                    "DL0 is unchanged in this circumstance."
                )
                self._r1_empty_warn = True
            return True
        else:
            return False

    def _check_dl0_empty(self, waveforms):
        if waveforms is None:
            if not self._dl0_empty_warn:
                warnings.warn(
                    "Encountered an event with no DL0 data. "
                    "DL1 is unchanged in this circumstance."
                )
                self._dl0_empty_warn = True
            return True
        else:
            return False

    def _calibrate_dl0(self, event, telid):
        waveforms = event.r1.tel[telid].waveform
        selected_gain_channel = event.r1.tel[telid].selected_gain_channel
        if self._check_r1_empty(waveforms):
            return

        reduced_waveforms_mask = self.data_volume_reducer(
            waveforms, telid=telid, selected_gain_channel=selected_gain_channel
        )

        waveforms_copy = waveforms.copy()
        waveforms_copy[~reduced_waveforms_mask] = 0
        event.dl0.tel[telid].waveform = waveforms_copy
        event.dl0.tel[telid].selected_gain_channel = selected_gain_channel

    def _calibrate_dl1(self, event, telid):
        waveforms = event.dl0.tel[telid].waveform
        selected_gain_channel = event.r1.tel[telid].selected_gain_channel
        if self._check_dl0_empty(waveforms):
            return
        n_pixels, n_samples = waveforms.shape
        if n_samples == 1:
            # To handle ASTRI and dst
            # TODO: Improved handling of ASTRI and dst
            #   - dst with custom EventSource?
            #   - Read into dl1 container directly?
            #   - Don't do anything if dl1 container already filled
            #   - Update on SST review decision
            charge = waveforms[..., 0].astype(np.float32)
            peak_time = np.zeros(n_pixels, dtype=np.float32)
        else:
            charge, peak_time = self.image_extractor(
                waveforms, telid=telid, selected_gain_channel=selected_gain_channel
            )

        # Calibrate extracted charge
        pedestal = event.calibration.tel[telid].dl1.pedestal_offset
        absolute = event.calibration.tel[telid].dl1.absolute_factor
        relative = event.calibration.tel[telid].dl1.relative_factor
        charge -= pedestal
        charge *= relative / absolute

        event.dl1.tel[telid].image = charge
        event.dl1.tel[telid].peak_time = peak_time

    def __call__(self, event):
        """
        Perform the full camera calibration from R1 to DL1. Any calibration
        relating to data levels before the data level the file is read into
        will be skipped.

        Parameters
        ----------
        event : container
            A `ctapipe` event container
        """
        # TODO: How to handle different calibrations depending on telid?
        tel = event.r1.tel or event.dl0.tel or event.dl1.tel
        for telid in tel.keys():
            self._calibrate_dl0(event, telid)
            self._calibrate_dl1(event, telid)
