"""
Definition of the `CameraCalibrator` class, providing all steps needed to apply
calibration and image extraction, as well as supporting algorithms.
"""

import warnings

import numpy as np
from astropy import units as u

from ctapipe.core import Component
from ctapipe.image.extractor import NeighborPeakWindowSum
from ctapipe.image.reducer import NullDataVolumeReducer

__all__ = ["CameraCalibrator"]


def integration_correction(
    reference_pulse_shape, reference_pulse_step, sample_width_ns,
    window_width, window_shift
):
    """
    Obtain the correction for the integration window specified.

    For any integration window applied to a noise-less unit pulse, the
    correction (returned by this function) multiplied by the integration
    result should equal 1.

    This correction therefore corrects for the Cherenkov signal that may be
    outside the integration window, and removes any dependence of the resulting
    image on the window_width and window_shift parameters. However, the width
    and shift of the window should still be optimised for the pulse finding and
    to minimise the noise included in the integration.

    Parameters
    ----------
    reference_pulse_shape : ndarray
        Numpy array containing the pulse shape for each gain channel
    reference_pulse_step : float
        The step in time for each sample of the reference pulse shape in ns
    sample_width_ns : float
        The width of the waveform sample time bin in ns
    window_width : int
        Width of the integration window (in units of n_samples)
    window_shift : int
        Shift to before the peak for the start of the integration window
        (in units of n_samples)

    Returns
    -------
    correction : ndarray
        Value of the integration correction for each gain channel
    """
    n_channels = len(reference_pulse_shape)
    correction = np.ones(n_channels, dtype=np.float)
    for ichannel, pulse_shape in enumerate(reference_pulse_shape):
        pulse_max_sample = pulse_shape.size * reference_pulse_step
        pulse_shape_x = np.arange(0, pulse_max_sample, reference_pulse_step)
        sampled_edges = np.arange(0, pulse_max_sample, sample_width_ns)

        sampled_pulse, _ = np.histogram(
            pulse_shape_x, sampled_edges, weights=pulse_shape, density=True
        )
        n_samples = sampled_pulse.size
        start = sampled_pulse.argmax() - window_shift
        start = start if start >= 0 else 0
        end = start + window_width
        end = end if end < n_samples else n_samples
        if start >= end:
            continue

        integration = sampled_pulse[start:end] * sample_width_ns
        correction[ichannel] = 1.0 / np.sum(integration)

    return correction


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
        **kwargs
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

        if data_volume_reducer is None:
            data_volume_reducer = NullDataVolumeReducer(parent=self)
        self.data_volume_reducer = data_volume_reducer

        if image_extractor is None:
            image_extractor = NeighborPeakWindowSum(parent=self, subarray=subarray)
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
            shift = self.image_extractor.window_shift.tel[None]
            width = self.image_extractor.window_width.tel[None]
            camera = self.subarray.tel[telid].camera
            shape = camera.readout.reference_pulse_shape
            step_ns = camera.readout.reference_pulse_step.to_value(u.ns)
            sample_width_ns = (1/camera.readout.sampling_rate).to_value(u.ns)
            correction = integration_correction(
                shape, step_ns, sample_width_ns, width, shift
            )
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
        if self._check_r1_empty(waveforms):
            return

        reduced_waveforms = self.data_volume_reducer(waveforms)
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
            charge = waveforms[..., 0]
            pulse_time = np.zeros(n_pixels)
        else:
            # TODO: apply timing correction to waveforms before charge extraction
            charge, pulse_time = self.image_extractor(waveforms, telid=telid)

            # Apply integration correction
            # TODO: Remove integration correction
            correction = self._get_correction(event, telid)
            charge = charge * correction

        # Calibrate extracted charge
        pedestal = event.calibration.tel[telid].dl1.pedestal_offset
        absolute = event.calibration.tel[telid].dl1.absolute_factor
        relative = event.calibration.tel[telid].dl1.relative_factor
        charge = (charge - pedestal) * relative / absolute

        event.dl1.tel[telid].image = charge
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
        tel = event.r1.tel or event.dl0.tel or event.dl1.tel
        for telid in tel.keys():
            self._calibrate_dl0(event, telid)
            self._calibrate_dl1(event, telid)
