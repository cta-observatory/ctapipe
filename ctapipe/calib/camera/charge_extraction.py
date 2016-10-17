from abc import abstractmethod
import numpy as np
from traitlets import Int, Unicode

from ctapipe.core import Component


def all_subclasses(cls):
    return cls.__subclasses__() + [g for s in cls.__subclasses__()
                                   for g in all_subclasses(s)]


class ChargeExtractor(Component):
    name = 'ChargeExtractor'

    def __init__(self, waveforms, parent, **kwargs):
        super().__init__(parent=parent, **kwargs)

        self.waveforms = waveforms
        self.nchan, self.npix, self.nsamples = waveforms.shape

    @abstractmethod
    def extract_charge(self):
        """ Calls the relevant functions to fully extract the charge for the
        particular extractor """


class Integrator(ChargeExtractor):
    def __init__(self, waveforms, parent, **kwargs):
        super().__init__(waveforms, parent=parent, **kwargs)

        self.__window_width = np.zeros((self.nchan, self.npix), dtype=np.intp)
        self.__window_start = np.zeros((self.nchan, self.npix), dtype=np.intp)

        self.integration_window = np.zeros_like(self.waveforms, dtype=bool)
        self.windowed_waveforms = None
        self.integrated_waveform = None
        self.peakpos = None

    @property
    def window_width(self):
        return self.__window_width

    @window_width.setter
    def window_width(self, ndarray):
        """
        Parameters
        ----------
        ndarray : ndarray
            Numpy array of dimensions (nchan*npix) containing the window
            size of the integration window for each pixel
        """
        ndarray = np.where(ndarray > self.nsamples, self.nsamples, ndarray)
        self.__window_width = ndarray

    @property
    def window_start(self):
        return self.__window_start

    @window_start.setter
    def window_start(self, ndarray):
        """
        Parameters
        ----------
        ndarray : ndarray
            Numpy array of dimensions (nchan*npix) containing the start
            position of the integration window for each pixel
        """
        width = self.window_width
        # noinspection PyTypeChecker
        ndarray = np.where(ndarray < 0, 0, ndarray)
        print(width.shape)
        print(self.nsamples)
        ndarray = np.where(ndarray + width > self.nsamples,
                           self.nsamples - width, ndarray)
        self.__window_start = ndarray

    def define_window(self):
        start = self.window_start
        end = start + self.window_width

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
        if not self.window_width.all():
            self.log.warn('all window_widths are zero')
        self.get_window_start()
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

    def __init__(self, waveforms, parent, **kwargs):
        super().__init__(waveforms, parent=parent, **kwargs)
        self.peakpos = [None, None]

    def get_window_width(self):
        self.window_width[:] = self.nsamples

    def get_window_start(self):
        self.window_start[:] = 0


class WindowIntegrator(Integrator):
    name = 'WindowIntegrator'
    window_width_arg = Int(7, help='Define the width of the integration '
                                   'window').tag(config=True)

    def __init__(self, waveforms, parent, **kwargs):
        super().__init__(waveforms, parent=parent, **kwargs)

    def get_window_width(self):
        self.window_width[:] = self.window_width_arg

    @abstractmethod
    def get_window_start(self):
        """"""


class SimpleIntegrator(WindowIntegrator):
    name = 'SimpleIntegrator'
    window_start_arg = Int(3, help='Define the start of the integration '
                                   'window').tag(config=True)

    def __init__(self, waveforms, parent, **kwargs):
        super().__init__(waveforms, parent=parent, **kwargs)
        self.peakpos = [None, None]

    def get_window_start(self):
        self.window_start[:] = self.window_start_arg


class PeakFindingIntegrator(WindowIntegrator):
    name = 'PeakFindingIntegrator'
    window_shift_arg = Int(3, help='Define the shift of the integration '
                                   'window from the peakpos '
                                   '(peakpos - shift').tag(config=True)
    sig_amp_cut_HG = Int(2, allow_none=True,
                         help='Define the cut above which a sample is '
                              'considered as significant for PeakFinding '
                              'in the HG channel').tag(config=True)
    sig_amp_cut_LG = Int(4, allow_none=True,
                         help='Define the cut above which a sample is '
                              'considered as significant for PeakFinding '
                              'in the LG channel').tag(config=True)

    def __init__(self, waveforms, parent, **kwargs):
        super().__init__(waveforms, parent=parent, **kwargs)
        self.sig_pixels = np.ones((self.nchan, self.npix))
        self.sig_channel = np.ones(self.nchan)
        self.significant_waveforms = self.waveforms
        self.window_shift = np.zeros((self.nchan, self.npix), dtype=np.intp)
        self.window_shift[:] = self.window_shift_arg

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
        print(self.peakpos.shape)
        print(self.window_shift.shape)
        self.window_start = self.peakpos - self.window_shift

    @abstractmethod
    def find_peak(self):
        """ Find the peak to define window around """


class GlobalPeakIntegrator(PeakFindingIntegrator):
    name = 'PeakFindingIntegrator'

    def __init__(self, waveforms, parent, **kwargs):
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

    def __init__(self, waveforms, parent, **kwargs):
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

    def __init__(self, waveforms, nei, parent, **kwargs):
        super().__init__(waveforms, parent=parent, **kwargs)
        self.nei = nei

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
    # noinspection PyTypeChecker
    subclasses = all_subclasses(ChargeExtractor)
    subclass_names = [c.__name__ for c in subclasses]

    extractor = Unicode('NeighbourPeakIntegrator',
                        help='Charge extraction scheme to use: {}'
                        .format(subclass_names)).tag(config=True)

    # TODO: temp extractor argument while factory classes are being defined
    def __init__(self, extractor=None, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)

        if extractor:
            self.extractor = extractor

    def get_extractor(self):
        for c in self.subclasses:
            if c.__name__ == self.extractor:
                return c
