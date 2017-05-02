"""
Waveform cleaning algorithms (smoothing, filtering, baseline subtraction)
"""

from traitlets import Int, CaselessStrEnum
from ctapipe.core import Component, Factory
import numpy as np
from scipy.signal import general_gaussian
from abc import abstractmethod

__all__ = ['WaveformCleanerFactory', 'CHECMWaveformCleaner',
           'NullWaveformCleaner']


class WaveformCleaner(Component):
    name = 'WaveformCleaner'

    def __init__(self, config, tool, **kwargs):
        """
        Base component to handle the cleaning of the waveforms in an image.

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
        kwargs
        """
        super().__init__(config=config, parent=tool, **kwargs)

    @abstractmethod
    def apply(self, waveforms):
        """
        Apply the cleaning method to the waveforms
        
        Parameters
        ----------
        waveforms : ndarray
            Waveforms stored in a numpy array of shape
            (n_chan, n_pix, n_samples).

        Returns
        -------
        cleaned : ndarray
            Cleaned waveforms stored in a numpy array of shape
            (n_chan, n_pix, n_samples).

        """


class NullWaveformCleaner(WaveformCleaner):
    """
    Dummy waveform cleaner that simply returns its input
    """
    name = 'NullWaveformCleaner'

    def apply(self, waveforms):
        return waveforms


class CHECMWaveformCleaner(WaveformCleaner):
    name = 'CHECMWaveformCleaner'

    window_width = Int(16, help='Define the width of the pulse '
                                'window').tag(config=True)
    window_shift = Int(8, help='Define the shift of the pulse window from the '
                               'peakpos (peakpos - shift).').tag(config=True)
    t0 = Int(None, allow_none=True,
             help='Override the value of t0').tag(config=True)

    def __init__(self, config, tool, **kwargs):
        """
        Waveform cleaner used by CHEC-M.
        
        This cleaner performs 2 basline subtractions: a simple subtraction
        using the average of the first 32 samples in the waveforms, then a 
        convolved baseline subtraction to remove and low frequency drifts in 
        the baseline.

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
        kwargs
        """
        super().__init__(config=config, tool=tool, **kwargs)

        # Cleaning steps for plotting
        self.stages = {}

        if self.t0:
            self.log.info("User has set t0, extracted t0 will be overridden")

        self.kernel = general_gaussian(10, p=1.0, sig=32)

    def apply(self, waveforms):
        samples = waveforms[0]
        npix, nsamples = samples.shape

        # Subtract initial baseline
        baseline_sub = samples - np.mean(samples[:, :32], axis=1)[:, None]

        # Get average waveform and define t0 for pulse window
        avg_wf = np.mean(baseline_sub, axis=0)
        t0 = np.argmax(avg_wf)
        if self.t0:
            t0 = self.t0

        # Set Windows
        pw_l = t0 - self.window_shift
        pw_r = pw_l + self.window_width
        if pw_l < 0:
            pw_l = 0
        if pw_r >= nsamples:
            pw_r = nsamples - 1
        pulse_window = np.s_[pw_l:pw_r]

        # Define the waveform without the pulse
        no_pulse = np.ma.array(baseline_sub, mask=False, fill_value=0)
        no_pulse.mask[:, pulse_window] = True
        no_pulse = np.ma.filled(no_pulse)

        # Get smooth baseline (no pulse)
        smooth_flat = np.convolve(no_pulse.ravel(), self.kernel, "same")
        smooth_baseline = np.reshape(smooth_flat, samples.shape)
        no_pulse_std = np.std(no_pulse, axis=1)
        smooth_baseline_std = np.std(smooth_baseline, axis=1)
        smooth_baseline *= (no_pulse_std / smooth_baseline_std)[:, None]

        # Get smooth waveform
        smooth_wf = baseline_sub  # self.wf_smoother.apply(baseline_sub)

        # Subtract smooth baseline
        cleaned = smooth_wf - smooth_baseline

        self.stages['0: raw'] = samples
        self.stages['1: baseline_sub'] = baseline_sub
        self.stages['2: avg_wf'] = avg_wf
        self.stages['t0'] = t0
        self.stages['window_start'] = pw_l
        self.stages['window_end'] = pw_r
        self.stages['3: no_pulse'] = no_pulse
        self.stages['4: smooth_baseline'] = smooth_baseline
        self.stages['5: smooth_wf'] = smooth_wf
        self.stages['6: cleaned'] = cleaned

        return cleaned[None, :]


class WaveformCleanerFactory(Factory):
    """
    Factory to obtain a WaveformCleaner.
    """
    name = "WaveformCleanerFactory"
    description = "Obtain WavefromCleaner based on cleaner traitlet"

    subclasses = Factory.child_subclasses(WaveformCleaner)
    subclass_names = [c.__name__ for c in subclasses]

    cleaner = CaselessStrEnum(subclass_names, 'NullWaveformCleaner',
                              help='Waveform cleaning method to '
                                   'use.').tag(config=True)

    # Product classes traits
    window_width = Int(16, help='Define the width of the pulse '
                                'window').tag(config=True)
    window_shift = Int(8, help='Define the shift of the pulse window from the '
                               'peakpos (peakpos - shift).').tag(config=True)
    t0 = Int(None, allow_none=True,
             help='Override the value of t0').tag(config=True)

    def get_factory_name(self):
        return self.name

    def get_product_name(self):
        return self.cleaner
