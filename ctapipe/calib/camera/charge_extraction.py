from abc import abstractmethod, ABCMeta
import numpy as np
from traitlets import Int, Unicode

from ctapipe.core import Component


def all_subclasses(cls):
    return cls.__subclasses__() + [g for s in cls.__subclasses__()
                                   for g in all_subclasses(s)]


class AbstractMeta(type(Component), ABCMeta):
    """Class to allow @abstractmethod to work"""
    # TODO: remove once Component is made abstract


class ChargeExtractor(Component, metaclass=AbstractMeta):
    name = 'ChargeExtractor'

    def __init__(self, waveforms, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)

        self.waveforms = waveforms
        self.nchan, self.npix, self.nsamples = waveforms.shape

    @staticmethod
    def require_neighbour():
        return False

    @abstractmethod
    def extract_charge(self):
        """ Calls the relevant functions to fully extract the charge for the
        particular extractor """


class Integrator(ChargeExtractor):
    name = 'Integrator'

    def __init__(self, waveforms, parent=None, **kwargs):
        super().__init__(waveforms, parent=parent, **kwargs)

        self.w_width = np.zeros((self.nchan, self.npix), dtype=np.intp)
        self.w_start = np.zeros((self.nchan, self.npix), dtype=np.intp)

        self.integration_window = np.zeros_like(self.waveforms, dtype=bool)
        self.windowed_waveforms = None
        self.integrated_waveform = None
        self.peakpos = None

    def check_window_width_and_start(self):
        width = self.w_width
        start = self.w_start
        if width is None:
            raise ValueError('window width has not been set')
        if start is None:
            raise ValueError('window start has not been set')
        if not width.all():
            self.log.warn('all window_widths are zero')

        width[width > self.nsamples] = self.nsamples
        start[start < 0] = 0
        sum_check = start + width > self.nsamples
        start[sum_check] = self.nsamples - width[sum_check]

    def define_window(self):
        start = self.w_start
        end = start + self.w_width

        # Obtain integration window using the indices of the waveforms array
        ind = np.indices((self.nchan, self.npix, self.nsamples))[2]
        self.integration_window = (ind >= start[..., None]) *\
                                  (ind < end[..., None])

        self.windowed_waveforms = np.ma.array(self.waveforms,
                                              mask=~self.integration_window)

    def integrate(self):
        integration = np.round(self.windowed_waveforms.sum(2)).astype(np.int)
        self.integrated_waveform = integration

    def extract_charge(self):
        self.get_window_width()
        self.get_window_start()
        self.check_window_width_and_start()
        self.define_window()
        self.integrate()
        return self.integrated_waveform, self.integration_window, self.peakpos

    @abstractmethod
    def get_window_width(self):
        """Get the width of the integration window"""

    @abstractmethod
    def get_window_start(self):
        """Get the starting point for the integration window"""


class FullIntegrator(Integrator):
    name = 'FullIntegrator'

    def __init__(self, waveforms, parent=None, **kwargs):
        super().__init__(waveforms, parent=parent, **kwargs)
        self.peakpos = [None, None]

    def get_window_width(self):
        self.w_width = np.full((3, 5), self.nsamples)

    def get_window_start(self):
        self.w_start[:] = 0


class WindowIntegrator(Integrator):
    name = 'WindowIntegrator'
    window_width = Int(7, help='Define the width of the integration '
                               'window').tag(config=True)

    def __init__(self, waveforms, parent, **kwargs):
        super().__init__(waveforms, parent=parent, **kwargs)

    def get_window_width(self):
        self.w_width[:] = self.window_width

    @abstractmethod
    def get_window_start(self):
        """"""


class SimpleIntegrator(WindowIntegrator):
    name = 'SimpleIntegrator'
    window_start = Int(3, help='Define the start of the integration '
                               'window').tag(config=True)

    def __init__(self, waveforms, parent=None, **kwargs):
        super().__init__(waveforms, parent=parent, **kwargs)
        self.peakpos = [None, None]

    def get_window_start(self):
        self.w_start[:] = self.window_start


class PeakFindingIntegrator(WindowIntegrator):
    name = 'PeakFindingIntegrator'
    window_shift = Int(3, help='Define the shift of the integration window '
                               'from the peakpos '
                               '(peakpos - shift').tag(config=True)
    sig_amp_cut_HG = Int(2, allow_none=True,
                         help='Define the cut above which a sample is '
                              'considered as significant for PeakFinding '
                              'in the HG channel').tag(config=True)
    sig_amp_cut_LG = Int(4, allow_none=True,
                         help='Define the cut above which a sample is '
                              'considered as significant for PeakFinding '
                              'in the LG channel').tag(config=True)

    def __init__(self, waveforms, parent=None, **kwargs):
        super().__init__(waveforms, parent=parent, **kwargs)
        self.sig_pixels = np.ones((self.nchan, self.npix))
        self.sig_channel = np.ones(self.nchan)
        self.significant_waveforms = self.waveforms
        self.w_shift = np.zeros((self.nchan, self.npix), dtype=np.intp)
        self.w_shift[:] = self.window_shift

    # Extract significant entries
    def extract_significant_entries(self):
        sig_entries = np.ones_like(self.waveforms, dtype=bool)
        if self.sig_amp_cut_HG:
            sig_entries[0] = self.waveforms[0] > self.sig_amp_cut_HG
        if self.nchan > 1 and self.sig_amp_cut_LG:
            sig_entries[1] = self.waveforms[1] > self.sig_amp_cut_LG
        self.sig_pixels = np.any(sig_entries, axis=2)
        self.sig_channel = np.any(self.sig_pixels, axis=1)
        if not self.sig_channel[0]:
            self.log.error("sigamp value excludes all values in HG channel")
        self.significant_waveforms = self.waveforms * sig_entries

    def get_window_start(self):
        if self.sig_amp_cut_HG or self.sig_amp_cut_HG:
            self.extract_significant_entries()
        self.find_peak()
        if self.peakpos is None:
            raise ValueError('peakpos has not been set')
        self.w_start = self.peakpos - self.w_shift

    @abstractmethod
    def find_peak(self):
        """ Find the peak to define window around """


class GlobalPeakIntegrator(PeakFindingIntegrator):
    name = 'PeakFindingIntegrator'

    def __init__(self, waveforms, parent=None, **kwargs):
        super().__init__(waveforms, parent=parent, **kwargs)

    def find_peak(self):
        max_time = self.significant_waveforms.argmax(2)
        max_sample = self.significant_waveforms.max(2)
        self.peakpos = np.zeros((self.nchan, self.npix), dtype=np.int)
        self.peakpos[0, :] = np.round(
            np.average(max_time[0], weights=max_sample[0]))
        if self.nchan > 1:
            if self.sig_channel[1]:
                self.peakpos[1, :] = np.round(
                    np.average(max_time[1], weights=max_sample[1]))
            else:
                self.log.info("LG not significant, using HG for peak finding "
                              "instead")
                self.peakpos[1, :] = self.peakpos[0]


class LocalPeakIntegrator(PeakFindingIntegrator):
    name = 'LocalPeakIntegrator'

    def __init__(self, waveforms, parent=None, **kwargs):
        super().__init__(waveforms, parent=parent, **kwargs)

    def find_peak(self):
        self.peakpos = self.significant_waveforms.argmax(2)
        if self.nchan > 1:  # If the LG is not significant, use the HG peakpos
            self.peakpos[1] = np.where(self.sig_pixels[1] < self.sig_pixels[0],
                                       self.peakpos[0], self.peakpos[1])


class NeighbourPeakIntegrator(PeakFindingIntegrator):
    name = 'NeighbourPeakIntegrator'
    lwt = Int(0, help='Weight of the local pixel (0: peak from neighbours '
                      'only, 1: local pixel counts as much '
                      'as any neighbour').tag(config=True)

    def __init__(self, waveforms, nei=None, parent=None, **kwargs):
        super().__init__(waveforms, parent=parent, **kwargs)
        self.nei = nei

    @staticmethod
    def require_neighbour():
        return True

    def find_peak(self):
        self.peakpos = np.zeros((self.nchan, self.npix), dtype=np.int)
        sig_wav = self.significant_waveforms
        max_num_nei = len(max(self.nei, key=len))
        allvals = np.zeros((self.nchan, self.npix,
                            max_num_nei + 1, self.nsamples))
        for ipix, neighbours in enumerate(self.nei):
            num_nei = len(neighbours)
            allvals[:, ipix, :num_nei, :] = sig_wav[:, neighbours]
            allvals[:, ipix, num_nei, :] = sig_wav[:, ipix] * self.lwt
        sum_data = allvals.sum(2)
        self.peakpos = sum_data.argmax(2)


class ChargeExtractorFactory(Component):
    name = "ChargeExtractorFactory"
    description = "Obtain ChargeExtractor based on extractor traitlet"

    # noinspection PyTypeChecker
    subclasses = all_subclasses(ChargeExtractor)
    subclass_names = [c.__name__ for c in subclasses]

    extractor = Unicode('NeighbourPeakIntegrator',
                        help='Charge extraction scheme to use: {}'
                        .format(subclass_names)).tag(config=True)

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.product = None

    def get_product_name(self):
        return self.extractor

    def init_product(self, product_name=None):
        if not product_name:
            product_name = self.get_product_name()
        for subclass in self.subclasses:
            if subclass.__name__ == product_name:
                self.product = subclass
                return subclass
        raise KeyError('No subclass exists with name: '
                       '{}'.format(self.get_product_name()))

    def get_product(self, waveforms, nei, parent=None, config=None, **kwargs):
        if not self.product:
            self.init_product()
        product = self.product
        object_instance = product(waveforms, nei=nei,
                                  parent=parent, config=config, **kwargs)
        return object_instance
