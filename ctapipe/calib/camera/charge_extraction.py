from abc import abstractmethod
import numpy as np
from traitlets import Int, Unicode

from ctapipe.core import Component
from ctapipe.calib.camera.factory_proposal import Factory


class ChargeExtractor(Component):
    name = 'ChargeExtractor'

    def __init__(self, config, tool, **kwargs):
        """
        Base component for charge extractors

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
        super().__init__(config=config, parent=tool, **kwargs)

        # self._waveforms = None
        self._nchan = None
        self._npix = None
        self._nsamples = None

        self.peakpos = None
        self.neighbours = None

    @staticmethod
    def requires_neighbours():
        return False

    def check_neighbour_set(self):
        if self.requires_neighbours():
            if self.neighbours is None:
                self.log.exception("neighbours attribute must be set")
                raise ValueError()

    @abstractmethod
    def extract_charge(self, waveforms):
        """ Calls the relevant functions to fully extract the charge for the
        particular extractor """


class Integrator(ChargeExtractor):
    name = 'Integrator'

    def __init__(self, config, tool, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)

        self.integration_window = None

    def _check_window_width_and_start(self, width, start):
        if width is None:
            raise ValueError('window width has not been set')
        if start is None:
            raise ValueError('window start has not been set')
        if not width.all():
            self.log.warn('all window_widths are zero')

        width[width > self._nsamples] = self._nsamples
        start[start < 0] = 0
        print(start)
        print(width)
        print(self._nsamples)
        sum_check = start + width > self._nsamples
        start[sum_check] = self._nsamples - width[sum_check]

    def _define_window(self, start, width):
        end = start + width

        # Obtain integration window using the indices of the waveforms array
        ind = np.indices((self._nchan, self._npix, self._nsamples))[2]
        integration_window = (ind >= start[..., None]) * (ind < end[..., None])
        return integration_window

    @staticmethod
    def _window_waveforms(waveforms, window):
        windowed = np.ma.array(waveforms, mask=~window)
        return windowed

    @staticmethod
    def _integrate(windowed_waveforms):
        integration = np.round(windowed_waveforms.sum(2)).astype(np.int)
        return integration

    def extract_charge(self, waveforms):
        self.check_neighbour_set()

        self._nchan, self._npix, self._nsamples = waveforms.shape
        w_width = np.zeros((self._nchan, self._npix), dtype=np.intp)
        w_start = np.zeros((self._nchan, self._npix), dtype=np.intp)

        self._get_window_width(w_width)
        self._get_window_start(w_start, waveforms)
        self._check_window_width_and_start(w_width, w_start)
        window = self._define_window(w_start, w_width)
        windowed_waveforms = self._window_waveforms(waveforms, window)
        charge = self._integrate(windowed_waveforms)

        self.integration_window = window
        return charge

    @abstractmethod
    def _get_window_width(self, w_width):
        """Get the width of the integration window"""

    @abstractmethod
    def _get_window_start(self, w_start, waveforms):
        """Get the starting point for the integration window"""


class FullIntegrator(Integrator):
    name = 'FullIntegrator'

    def __init__(self, config, tool, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)

    def _get_window_width(self, w_width):
        w_width[:] = self._nsamples

    def _get_window_start(self, w_start, waveforms):
        w_start[:] = 0


class WindowIntegrator(Integrator):
    name = 'WindowIntegrator'
    window_width = Int(7, help='Define the width of the integration '
                               'window').tag(config=True)

    def __init__(self, config, tool, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)

    def _get_window_width(self, w_width):
        w_width[:] = self.window_width

    @abstractmethod
    def _get_window_start(self, w_start, waveforms):
        """"""


class SimpleIntegrator(WindowIntegrator):
    name = 'SimpleIntegrator'
    window_start = Int(3, help='Define the start of the integration '
                               'window').tag(config=True)

    def __init__(self, config, tool, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)

    def _get_window_start(self, w_start, waveforms):
        w_start[:] = self.window_start


class PeakFindingIntegrator(WindowIntegrator):
    name = 'PeakFindingIntegrator'
    window_shift = Int(3, help='Define the shift of the integration window '
                               'from the peakpos '
                               '(peakpos - shift)').tag(config=True)
    sig_amp_cut_HG = Int(2, allow_none=True,
                         help='Define the cut above which a sample is '
                              'considered as significant for PeakFinding '
                              'in the HG channel').tag(config=True)
    sig_amp_cut_LG = Int(4, allow_none=True,
                         help='Define the cut above which a sample is '
                              'considered as significant for PeakFinding '
                              'in the LG channel').tag(config=True)

    def __init__(self, config, tool, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)
        self._sig_channel = None
        self._sig_pixels = None
        # self.significant_waveforms = self.waveforms
        # self.w_shift = np.zeros((self.nchan, self.npix), dtype=np.intp)
        # self.w_shift[:] = self.window_shift

    # Extract significant entries
    def _extract_significant_entries(self, waveforms):
        sig_entries = np.ones(waveforms.shape, dtype=bool)
        if self.sig_amp_cut_HG:
            sig_entries[0] = waveforms[0] > self.sig_amp_cut_HG
        if self._nchan > 1 and self.sig_amp_cut_LG:
            sig_entries[1] = waveforms[1] > self.sig_amp_cut_LG
        self._sig_pixels = np.any(sig_entries, axis=2)
        self._sig_channel = np.any(self._sig_pixels, axis=1)
        if not self._sig_channel[0]:
            self.log.error("sigamp value excludes all values in HG channel")
        return waveforms * sig_entries

    def _get_window_start(self, w_start, waveforms):
        significant_samples = waveforms
        if self.sig_amp_cut_HG or self.sig_amp_cut_HG:
            significant_samples = self._extract_significant_entries(waveforms)

        peakpos = np.zeros((self._nchan, self._npix), dtype=np.int)
        self._find_peak(significant_samples, peakpos)

        w_start[:] = peakpos - self.window_shift
        self.peakpos = peakpos

    @abstractmethod
    def _find_peak(self, significant_samples, peakpos):
        """ Find the peak to define window around """


class GlobalPeakIntegrator(PeakFindingIntegrator):
    name = 'PeakFindingIntegrator'

    def __init__(self, config, tool, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)

    def _find_peak(self, significant_samples, peakpos):
        max_t = significant_samples.argmax(2)
        max_s = significant_samples.max(2)

        peakpos[0, :] = np.round(np.average(max_t[0], weights=max_s[0]))
        if self._nchan > 1:
            if self._sig_channel[1]:
                peakpos[1, :] = np.round(
                    np.average(max_t[1], weights=max_s[1]))
            else:
                self.log.info("LG not significant, using HG for peak finding "
                              "instead")
                peakpos[1, :] = peakpos[0]


class LocalPeakIntegrator(PeakFindingIntegrator):
    name = 'LocalPeakIntegrator'

    def __init__(self, config, tool, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)

    def _find_peak(self, significant_samples, peakpos):
        peakpos[:] = significant_samples.argmax(2)
        sig_pix = self._sig_pixels
        if self._nchan > 1:  # If the LG is not significant, use the HG peakpos
            peakpos[1] = np.where(sig_pix[1] < sig_pix[0],
                                  peakpos[0], peakpos[1])


class NeighbourPeakIntegrator(PeakFindingIntegrator):
    name = 'NeighbourPeakIntegrator'
    lwt = Int(0, help='Weight of the local pixel (0: peak from neighbours '
                      'only, 1: local pixel counts as much '
                      'as any neighbour').tag(config=True)

    def __init__(self, config, tool, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)
        self.neighbours = None

    @staticmethod
    def require_neighbour():
        return True

    def _find_peak(self, significant_samples, peakpos):
        sig_sam = significant_samples
        max_num_nei = len(max(self.neighbours, key=len))
        allvals = np.zeros((self._nchan, self._npix,
                            max_num_nei + 1, self._nsamples))
        for ipix, neighbours in enumerate(self.neighbours):
            num_nei = len(neighbours)
            allvals[:, ipix, :num_nei, :] = sig_sam[:, neighbours]
            allvals[:, ipix, num_nei, :] = sig_sam[:, ipix] * self.lwt
        sum_data = allvals.sum(2)
        peakpos[:] = sum_data.argmax(2)


class ChargeExtractorFactory(Factory):
    name = "ChargeExtractorFactory"
    description = "Obtain ChargeExtractor based on extractor traitlet"

    subclasses = Factory.child_subclasses(ChargeExtractor)
    subclass_names = [c.__name__ for c in subclasses]

    extractor = Unicode('NeighbourPeakIntegrator',
                        help='Charge extraction scheme to use: {}'
                        .format(subclass_names)).tag(config=True)

    # Product classes traits
    # Would be nice to have these automatically set...!
    window_width = Int(7, help='Define the width of the integration '
                               'window. Only applicable to '
                               'WindowIntegrators.').tag(config=True)
    window_start = Int(3, help='Define the start of the integration '
                               'window. Only applicable to '
                               'SimpleIntegrators.').tag(config=True)
    window_shift = Int(3, help='Define the shift of the integration window '
                               'from the peakpos (peakpos - shift). Only '
                               'applicable to '
                               'PeakFindingIntegrators.').tag(config=True)
    sig_amp_cut_HG = Int(2, allow_none=True,
                         help='Define the cut above which a sample is '
                              'considered as significant for PeakFinding '
                              'in the HG channel. Only applicable to '
                              'PeakFindingIntegrators.').tag(config=True)
    sig_amp_cut_LG = Int(4, allow_none=True,
                         help='Define the cut above which a sample is '
                              'considered as significant for PeakFinding '
                              'in the LG channel. Only applicable to '
                              'PeakFindingIntegrators.').tag(config=True)

    def get_factory_name(self):
        return self.name

    def get_product_name(self):
        return self.extractor
