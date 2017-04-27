"""
Module containing general functions that will perform the dl1 calibration
on any event regardless of the origin/telescope, and store the calibration
inside the event container.
"""
import numpy as np
from .charge_extractors import NeighbourPeakIntegrator
from .waveform_cleaning import NullWaveformCleaner
from ctapipe.core import Component
from ctapipe.instrument import CameraGeometry
from ctapipe.core.traits import Float, Bool


def integration_correction(event, telid, window_width, window_shift):
    """
    Obtain the integration correction for the window specified.

    This correction accounts for the cherenkov signal that may be missed due
    to a smaller integration window by looking at the reference pulse shape.

    Provides the same result as set_integration_correction from readhess.

    Parameters
    ----------
    event : container
        A `ctapipe` event container
    telid : int
        telescope id
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
    n_chan = event.inst.num_channels[telid]
    correction = [1] * n_chan
    for chan in range(n_chan):

        shape = event.mc.tel[telid].reference_pulse_shape[chan]
        step = event.mc.tel[telid].meta['refstep']
        time_slice = event.mc.tel[telid].time_slice

        if shape.all() is False or time_slice == 0 or step == 0:
            continue

        ref_x = np.arange(0, shape.size * step, step)
        edges = np.arange(0, shape.size * step + 1, time_slice)

        sampled = np.histogram(ref_x, edges, weights=shape, density=True)[0]
        n_samples = sampled.size
        start = sampled.argmax() - window_shift
        end = start + window_width

        if window_width > n_samples:
            window_width = n_samples
        if start < 0:
            start = 0
        if start + window_width > n_samples:
            start = n_samples - window_width

        correction[chan] = 1 / sampled[start:end].sum()

    return correction


class CameraDL1Calibrator(Component):
    name = 'CameraCalibrator'
    radius = Float(None, allow_none=True,
                   help='Pixels within radius from a pixel are considered '
                        'neighbours to the pixel. Set to None for the default '
                        '(1.4 * min_pixel_seperation).').tag(config=True)
    correction = Bool(True,
                      help='Apply an integration correction to the charge to '
                           'account for the full cherenkov signal that your '
                           'smaller integration window may be '
                           'missing.').tag(config=True)
    clip_amplitude = Float(None, allow_none=True,
                           help='Amplitude in p.e. above which the signal is '
                                'clipped. Set to None for no '
                                'clipping.').tag(config=True)

    def __init__(self, config, tool, extractor=None, cleaner=None, **kwargs):
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
        tool : ctapipe.core.Tool
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
        super().__init__(config=config, parent=tool, **kwargs)
        self.cleaner = cleaner
        self.extractor = extractor
        if self.extractor is None:
            self.extractor = NeighbourPeakIntegrator(config, tool)
        self.cleaner = cleaner
        if self.cleaner is None:
            self.cleaner = NullWaveformCleaner(config, tool)
        self._current_url = None
        self._dl0_empty_warn = False

        self.neighbour_dict = {}
        self.correction_dict = {}

    def _check_url_change(self, event):
        """
        Check if the event comes from a different file to the previous events.
        If it has, then the neighbour and correction dicts need to be reset
        as telescope ids might not indicate the same telescope type as before.

        Parameters
        ----------
        event : container
            A `ctapipe` event container
        """
        if 'input' in event.meta:
            url = event.meta['input']
            if not self._current_url:
                self._current_url = url
            if url != self._current_url:
                self.log.warning("A new CameraDL1Calibrator should be created"
                                 "for each individual file so stored "
                                 "neighbours and integration_correction "
                                 "match the correct telid")
                self.neighbour_dict = {}
                self.correction_dict = {}

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
            True if dl0.tel[telid].pe_samples is not None, else false.
        """
        dl0 = event.dl0.tel[telid].pe_samples
        if dl0 is not None:
            return True
        else:
            if not self._dl0_empty_warn:
                self.log.warning("Encountered an event with no DL0 data. "
                                 "DL1 is unchanged in this circumstance.")
                self._dl0_empty_warn = True
            return False

    def get_geometry(self, event, telid):
        """
        Obtain the neighbouring pixels for this telescope.

        Parameters
        ----------
        event : container
            A `ctapipe` event container
        telid : int
            The telescope id.
            The neighbours are calculated once per telescope.
        """
        return CameraGeometry.guess(*event.inst.pixel_pos[telid],
                                    event.inst.optical_foclen[telid])

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
        """
        if telid in self.correction_dict:
            return self.correction_dict[telid]
        else:
            try:
                shift = self.extractor.window_shift
                width = self.extractor.window_width
                self.correction_dict[telid] = \
                    integration_correction(event, telid, width, shift)
                return self.correction_dict[telid]
            except AttributeError:
                # Don't apply correction when window_shift or window_width
                # does not exist in extractor, or when container does not have
                # a reference pulse shape
                return 1

    def calibrate(self, event):
        """
        Fill the dl1 container with the calibration data that results from the
        configuration of this calibrator.

        Parameters
        ----------
        event : container
            A `ctapipe` event container
        """
        self._check_url_change(event)
        for telid in event.dl0.tels_with_data:

            if self.check_dl0_exists(event, telid):
                waveforms = event.dl0.tel[telid].pe_samples

                # Clean waveforms
                cleaned = self.cleaner.apply(waveforms)

                # Extract charge
                if self.extractor.requires_neighbours():
                    e = self.extractor
                    e.neighbours = self.get_geometry(event, telid).neighbors
                extract = self.extractor.extract_charge
                charge, peakpos, window = extract(cleaned)

                # Apply integration correction
                if self.correction:
                    corrected = charge * self.get_correction(event, telid)
                else:
                    corrected = charge

                # Clip amplitude
                if self.clip_amplitude:
                    corrected[corrected > self.clip_amplitude] = \
                        self.clip_amplitude

                # Store into event container
                event.dl1.tel[telid].image = corrected
                event.dl1.tel[telid].extracted_samples = window
                event.dl1.tel[telid].peakpos = peakpos
                event.dl1.tel[telid].cleaned = cleaned
