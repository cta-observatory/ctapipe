import numpy as np
from ctapipe.core import Component
from ctapipe.calib.camera.gainselection import ManualGainSelector
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
                 gain_selector=None,
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
        gain_selector : ctapipe.calib.camera.gainselection.GainSelector
            The GainSelector to use. If None, then ManualGainSelector will be
            used, which by default selects the high/first gain channel.
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

        if gain_selector is None:
            gain_selector = ManualGainSelector(parent=self)
        self.gain_selector = gain_selector

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
            selected_gain_channel = event.r1.tel[telid].selected_gain_channel
            shift = self.image_extractor.window_shift
            width = self.image_extractor.window_width
            shape = event.mc.tel[telid].reference_pulse_shape
            n_chan = shape.shape[0]
            step = event.mc.tel[telid].meta['refstep']
            time_slice = event.mc.tel[telid].time_slice
            correction = integration_correction(n_chan, shape, step,
                                                time_slice, width, shift)
            pixel_correction = correction[selected_gain_channel]
            return pixel_correction
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

        # Perform gain selection. This is typically not the responsibility of
        # ctapipe; DL0 (and R1) waveforms are aleady gain selected and
        # therefore single channel. However, the waveforms read from
        # simtelarray do not have the gain selection applied, and so must be
        # done as part of the calibration step to ensure the correct
        # waveform dimensions.
        waveforms_gs, selected_gain_channel = self.gain_selector(waveforms)
        if selected_gain_channel is not None:
            event.r1.tel[telid].selected_gain_channel = selected_gain_channel
        else:
            # If pixel_channel is None, then waveforms has already been
            # pre-gainselected, and presumably the selected_gain_channel
            # container is filled by the EventSource
            if event.r1.tel[telid].selected_gain_channel is None:
                raise ValueError(
                    "EventSource is loading pre-gainselected waveforms "
                    "without filling the selected_gain_channel container"
                )

        reduced_waveforms = self.data_volume_reducer(waveforms_gs)
        event.dl0.tel[telid].waveform = reduced_waveforms

    def _calibrate_dl1(self, event, telid):
        waveforms = event.dl0.tel[telid].waveform
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
            corrected_charge = waveforms[..., 0]
            pulse_time = np.zeros(n_pixels)
        else:
            # TODO: pass camera to ImageExtractor.__init__
            if self.image_extractor.requires_neighbors():
                camera = event.inst.subarray.tel[telid].camera
                self.image_extractor.neighbors = camera.neighbor_matrix_where
            # TODO: apply timing correction to waveforms before charge extraction
            charge, pulse_time = self.image_extractor(waveforms)

            # Apply integration correction
            # TODO: Remove integration correction
            correction = self._get_correction(event, telid)
            corrected_charge = charge * correction

        # Calibrate extracted charge
        pedestal = event.mon.tel[telid].dl1.pedestal
        absolute = event.mon.tel[telid].dl1.absolute
        relative = event.mon.tel[telid].dl1.relative
        calibrated_charge = (corrected_charge - pedestal) * relative / absolute

        event.dl1.tel[telid].image = calibrated_charge
        event.dl1.tel[telid].pulse_time = pulse_time

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
