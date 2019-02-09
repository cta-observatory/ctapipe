"""
Calibrator for the DL0 -> DL1 data level transition.

This module handles the calibration from the DL0 data level to DL1. This
transition involves the waveform cleaning (such as filtering, smoothing,
or basline subtraction) performed by a cleaner from
`ctapipe.image.waveform_cleaning`, and the charge extraction technique
from `ctapipe.image.charge_extractors`.
"""
import numpy as np

from ...core import Component
from ...core.traits import Float
from ...image import NeighbourPeakIntegrator, NullWaveformCleaner

__all__ = ['CameraDL1Calibrator']


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

        sampled, se = np.histogram(ref_x, edges, weights=pshape, density=True)
        n_samples = sampled.size
        start = sampled.argmax() - window_shift
        end = start + window_width

        if window_width > n_samples:
            window_width = n_samples
        if start < 0:
            start = 0
        if start + window_width > n_samples:
            start = n_samples - window_width

        integration = np.diff(se)[start:end] * sampled[start:end]
        correction[chan] = 1 / np.sum(integration)

    return correction


class CameraDL1Calibrator(Component):
    """
    The calibrator for DL1 charge extraction. Fills the dl1 container.

    It handles the integration correction and, if required, the list of
    neighbours.

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
    extractor : ctapipe.calib.camera.charge_extractors.ChargeExtractor
        The extractor to use to extract the charge from the waveforms.
        By default the NeighbourPeakIntegrator with default configuration
        is used.
    cleaner : ctapipe.calib.camera.waveform_cleaners.Cleaner
        The waveform cleaner to use. By default no cleaning is
        applied to the waveforms.
    kwargs
    """
    radius = Float(None, allow_none=True,
                   help='Pixels within radius from a pixel are considered '
                        'neighbours to the pixel. Set to None for the default '
                        '(1.4 * min_pixel_seperation).').tag(config=True)
    clip_amplitude = Float(None, allow_none=True,
                           help='Amplitude in p.e. above which the signal is '
                                'clipped. Set to None for no '
                                'clipping.').tag(config=True)

    def __init__(self, config=None, tool=None, extractor=None, cleaner=None,
                 **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)
        self.extractor = extractor
        if self.extractor is None:
            self.extractor = NeighbourPeakIntegrator(config, tool)
        self.cleaner = cleaner
        if self.cleaner is None:
            self.cleaner = NullWaveformCleaner(config, tool)
        self._dl0_empty_warn = False

    def check_dl0_exists(self, event, telid):
        """
        Check that dl0 data exists. If it does not, then do not change dl1.

        This ensures that if the containers were filled from a file containing
        dl1 data, it is not overwritten by non-existant data.

        Parameters
        ----------
        event : container
            A `ctapipe` event container
        telid : int
            The telescope id.

        Returns
        -------
        bool
            True if dl0.tel[telid].waveform is not None, else false.
        """
        dl0 = event.dl0.tel[telid].waveform
        if dl0 is not None:
            return True
        else:
            if not self._dl0_empty_warn:
                self.log.warning("Encountered an event with no DL0 data. "
                                 "DL1 is unchanged in this circumstance.")
                self._dl0_empty_warn = True
            return False

    def get_correction(self, event, telid):
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
            shift = self.extractor.window_shift
            width = self.extractor.window_width
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

    def calibrate(self, event):
        """
        Fill the dl1 container with the calibration data that results from the
        configuration of this calibrator.

        Parameters
        ----------
        event : container
            A `ctapipe` event container
        """
        for telid in event.dl0.tels_with_data:

            if self.check_dl0_exists(event, telid):
                waveforms = event.dl0.tel[telid].waveform
                n_samples = waveforms.shape[2]
                if n_samples == 1:
                    # To handle ASTRI and dst
                    corrected = waveforms[..., 0]
                    window = np.ones(waveforms.shape)
                    peakpos = np.zeros(waveforms.shape[0:2])
                    cleaned = waveforms
                else:
                    # Clean waveforms
                    cleaned = self.cleaner.apply(waveforms)

                    # Extract charge
                    if self.extractor.requires_neighbours():
                        e = self.extractor
                        g = event.inst.subarray.tel[telid].camera
                        e.neighbours = g.neighbor_matrix_where
                    extract = self.extractor.extract_charge
                    charge, peakpos, window = extract(cleaned)

                    # Apply integration correction
                    correction = self.get_correction(event, telid)[:, None]
                    corrected = charge * correction

                # Clip amplitude
                if self.clip_amplitude:
                    corrected[corrected > self.clip_amplitude] = \
                        self.clip_amplitude

                # Store into event container
                event.dl1.tel[telid].image = corrected
                event.dl1.tel[telid].extracted_samples = window
                event.dl1.tel[telid].peakpos = peakpos
                event.dl1.tel[telid].cleaned = cleaned
