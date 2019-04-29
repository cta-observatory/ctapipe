import numpy as np
from ctapipe.core import Component
from ctapipe.image.reducer import NullDataVolumeReducer
from ctapipe.image.extractor import NeighborPeakWindowSum
import warnings

__all__ = ['CameraCalibrator']


def integration_correction(n_chan, pulse_shape, refstep, time_slice,
                           window_width, window_shift):
    """
    Obtain the integration correction for the window specified.

    This correction accounts for the cherenkov signal that may be missed due
    to a smaller integration window by looking at the reference pulse shape.

    Provides the same result as set_integration_correction from readhess.

    Parameters
    ----------
    n_chan : int
        Number of gain channels for the telescope
    pulse_shape : ndarray
        Numpy array containing the pulse shape for each channel.
    refstep : int
        The step in time for each sample of the reference pulse shape
    time_slice : int
        The step in time for each sample of the waveforms
    window_width : int
        Width of the integration window.
    window_shift : int
        Shift to before the peak for the start of the integration window.

    Returns
    -------
    correction : list[2]
        Value of the integration correction for this telescope for each
        channel.
    """
    correction = np.ones(n_chan)
    for chan in range(n_chan):
        pshape = pulse_shape[chan]
        if pshape.all() is False or time_slice == 0 or refstep == 0:
            continue

        ref_x = np.arange(0, pshape.size * refstep, refstep)
        edges = np.arange(0, pshape.size * refstep + 1, time_slice)

        sampled, sampled_edges = np.histogram(
            ref_x, edges, weights=pshape, density=True
        )
        n_samples = sampled.size
        start = sampled.argmax() - window_shift
        end = start + window_width

        if window_width > n_samples:
            window_width = n_samples
        if start < 0:
            start = 0
        if start + window_width > n_samples:
            start = n_samples - window_width

        integration = np.diff(sampled_edges)[start:end] * sampled[start:end]
        correction[chan] = 1 / np.sum(integration)

    return correction


class CameraCalibrator(Component):
    """
    Calibrator to handle the full camera calibration chain, in order to fill
    the DL1 data level in the event container.
    """
    def __init__(self, config=None, parent=None,
                 data_volume_reducer=None,
                 image_extractor=None,
                 **kwargs):
        """
        Parameters
        ----------
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
        kwargs
        """
        super().__init__(config=config, parent=parent, **kwargs)

        self._r1_empty_warn = False
        self._dl0_empty_warn = False

        if data_volume_reducer is None:
            data_volume_reducer = NullDataVolumeReducer(parent=self)
        self.data_volume_reducer = data_volume_reducer

        if image_extractor is None:
            image_extractor = NeighborPeakWindowSum(parent=self)
        self.image_extractor = image_extractor

    def _get_correction(self, event, telid):
        """
        Obtain the integration correction for this telescope.

        Parameters
        ----------
        event : container
            A `ctapipe` event container
        telid : int
            The telescope id.
            The integration correction is calculated once per telescope.

        Returns
        -------
        ndarray
        """
        try:
            shift = self.image_extractor.window_shift
            width = self.image_extractor.window_width
            shape = event.mc.tel[telid].reference_pulse_shape
            n_chan = shape.shape[0]
            step = event.mc.tel[telid].meta['refstep']
            time_slice = event.mc.tel[telid].time_slice
            correction = integration_correction(n_chan, shape, step,
                                                time_slice, width, shift)
            return correction
        except (AttributeError, KeyError):
            # Don't apply correction when window_shift or window_width
            # does not exist in extractor, or when container does not have
            # a reference pulse shape
            return np.ones(event.dl0.tel[telid].waveform.shape[0])

    def _check_r1_empty(self, waveforms):
        if waveforms is None:
            if not self._r1_empty_warn:
                warnings.warn("Encountered an event with no R1 data. "
                              "DL0 is unchanged in this circumstance.")
                self._r1_empty_warn = True
            return True
        else:
            return False

    def _check_dl0_empty(self, waveforms):
        if waveforms is None:
            if not self._dl0_empty_warn:
                warnings.warn("Encountered an event with no DL0 data. "
                              "DL1 is unchanged in this circumstance.")
                self._dl0_empty_warn = True
            return True
        else:
            return False

    def _calibrate_dl0(self, event, telid):
        waveforms = event.r1.tel[telid].waveform
        if self._check_r1_empty(waveforms):
            return
        # TODO: Add gain selection
        reduced_waveforms = self.data_volume_reducer(waveforms)
        event.dl0.tel[telid].waveform = reduced_waveforms

    def _calibrate_dl1(self, event, telid):
        waveforms = event.dl0.tel[telid].waveform
        if self._check_dl0_empty(waveforms):
            return
        n_samples = waveforms.shape[2]
        if n_samples == 1:
            # To handle ASTRI and dst
            # TODO: Improved handling of ASTRI and dst
            #   - dst with custom EventSource?
            #   - Read into dl1 container directly?
            #   - Don't do anything if dl1 container already filled
            #   - Update on SST review decision
            corrected_charge = waveforms[..., 0]
            pulse_time = np.zeros(waveforms.shape[0:2])
        else:
            # TODO: pass camera to ImageExtractor.__init__
            if self.image_extractor.requires_neighbors():
                camera = event.inst.subarray.tel[telid].camera
                self.image_extractor.neighbors = camera.neighbor_matrix_where
            charge, pulse_time = self.image_extractor(waveforms)

            # Apply integration correction
            # TODO: Remove integration correction
            correction = self._get_correction(event, telid)[:, None]
            corrected_charge = charge * correction

        event.dl1.tel[telid].image = corrected_charge
        event.dl1.tel[telid].pulse_time = pulse_time

        # TODO: Add charge calibration

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
        for telid in event.r1.tel.keys():
            self._calibrate_dl0(event, telid)
            self._calibrate_dl1(event, telid)
