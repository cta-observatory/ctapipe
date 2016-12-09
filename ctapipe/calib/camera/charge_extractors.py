from abc import abstractmethod
import numpy as np
from traitlets import Int, Unicode

from ctapipe.core import Component
from ctapipe.calib.camera.factory_proposal import Factory


class ChargeExtractor(Component):
    """
    Attributes
    ----------
    extracted_samples : ndarray
        Numpy array containing True where the samples lay within the
        integration window, and False where the samples lay outside. Has
        shape of (n_chan, n_pix, n_samples).
    peakpos : ndarray
        Numpy array of the peak position for each pixel. Has shape of
        (n_chan, n_pix).
    neighbours : list
        List of neighbours for each pixel. Changes per telescope.

    """
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

        self._nchan = None
        self._npix = None
        self._nsamples = None

        self.extracted_samples = None
        self.peakpos = None
        self.neighbours = None

    @staticmethod
    def requires_neighbours():
        """
        Method used for callers of the ChargeExtractor to know if the extractor
        requires knowledge of the pixel neighbours

        Returns
        -------
        bool
        """
        return False

    def check_neighbour_set(self):
        """
        Check if the pixel neighbours has been set for the extractor

        Raises
        -------
        ValueError
            If neighbours has not been set
        """
        if self.requires_neighbours():
            if self.neighbours is None:
                self.log.exception("neighbours attribute must be set")
                raise ValueError()

    @abstractmethod
    def extract_charge(self, waveforms):
        """
        Call the relevant functions to fully extract the charge for the
        particular extractor.

        Parameters
        ----------
        waveforms : ndarray
            Waveforms stored in a numpy array of shape
            (n_chan, n_pix, n_samples).

        Returns
        -------
        charge : ndarray
            Extracted charge stored in a numpy array of shape (n_chan, n_pix).
        """


class Integrator(ChargeExtractor):
    name = 'Integrator'

    def __init__(self, config, tool, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)

    def _check_window_width_and_start(self, width, start):
        """
        Check that the combination of window width and start positions fit
        within the readout window.

        Parameters
        ----------
        width : ndarray
            Numpy array containing the window width for each pixel. Shape =
            (n_chan, n_pix)
        start : ndarray
            Numpy array containing the window start for each pixel. Shape =
            (n_chan, n_pix)

        """
        # if width is None:
        #     raise ValueError('window width has not been set')
        # if start is None:
        #     raise ValueError('window start has not been set')
        if not width.all():
            self.log.warn('all window_widths are zero')

        width[width > self._nsamples] = self._nsamples
        start[start < 0] = 0
        sum_check = start + width > self._nsamples
        start[sum_check] = self._nsamples - width[sum_check]

    def _define_window(self, start, width):
        """
        Build the a numpy array of bools defining the integration window.

        Parameters
        ----------
        width : ndarray
            Numpy array containing the window width for each pixel. Shape =
            (n_chan, n_pix)
        start : ndarray
            Numpy array containing the window start for each pixel. Shape =
            (n_chan, n_pix)

        Returns
        -------
        integration_window : ndarray
            Numpy array containing True where the samples lay within the
            integration window, and False where the samples lay outside. Has
            shape of (n_chan, n_pix, n_samples).

        """
        end = start + width

        # Obtain integration window using the indices of the waveforms array
        ind = np.indices((self._nchan, self._npix, self._nsamples))[2]
        integration_window = (ind >= start[..., None]) * (ind < end[..., None])
        return integration_window

    @staticmethod
    def _window_waveforms(waveforms, window):
        """
        Mask the waveforms with the integration window

        Parameters
        ----------
        waveforms : ndarray
            Waveforms stored in a numpy array of shape
            (n_chan, n_pix, n_samples).
        window : ndarray
            Numpy array containing True where the samples lay within the
            integration window, and False where the samples lay outside. Has
            shape of (n_chan, n_pix, n_samples).

        Returns
        -------
        windowed : masked ndarray
            Waveforms masked with the integration window.

        """
        windowed = np.ma.array(waveforms, mask=~window)
        return windowed

    @staticmethod
    def _integrate(windowed_waveforms):
        """
        Integrate a the waveforms after they have been masked with the
        integration window.

        Parameters
        ----------
        windowed_waveforms : masked ndarray
            Waveforms array that has been masked by the integration_window
            array.

        Returns
        -------
        integration : ndarray
            Result of the integration. Has a shape of (n_chan, n_pix).

        """
        integration = windowed_waveforms.sum(2)
        return integration

    def extract_charge(self, waveforms):
        self.check_neighbour_set()
        self._nchan, self._npix, self._nsamples = waveforms.shape
        w_width = self._get_window_width()
        w_start = self._get_window_start(waveforms)
        self._check_window_width_and_start(w_width, w_start)
        window = self._define_window(w_start, w_width)
        windowed_waveforms = self._window_waveforms(waveforms, window)
        charge = self._integrate(windowed_waveforms)

        self.extracted_samples = window
        return charge

    @abstractmethod
    def _get_window_width(self):
        """
        Get the width of the integration window

        Returns
        -------
        width : ndarray
            Numpy array containing the window width for each pixel. Shape =
            (n_chan, n_pix)

        """

    @abstractmethod
    def _get_window_start(self, waveforms):
        """
        Get the starting point for the integration window

        Returns
        -------
        start : ndarray
            Numpy array containing the window start for each pixel. Shape =
            (n_chan, n_pix)

        """


class FullIntegrator(Integrator):
    name = 'FullIntegrator'

    def __init__(self, config, tool, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)

    def _get_window_width(self):
        return np.full((self._nchan, self._npix), self._nsamples,
                       dtype=np.intp)

    def _get_window_start(self, waveforms):
        return np.zeros((self._nchan, self._npix), dtype=np.intp)


class WindowIntegrator(Integrator):
    name = 'WindowIntegrator'
    window_width = Int(7, help='Define the width of the integration '
                               'window').tag(config=True)

    def __init__(self, config, tool, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)
        self.input_width = self.window_width

    def _get_window_width(self):
        return np.full((self._nchan, self._npix), self.window_width,
                       dtype=np.intp)

    @abstractmethod
    def _get_window_start(self, waveforms):
        """"""


class SimpleIntegrator(WindowIntegrator):
    name = 'SimpleIntegrator'
    window_start = Int(3, help='Define the start of the integration '
                               'window').tag(config=True)

    def __init__(self, config, tool, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)
        self.input_shift = self.window_start

    def _get_window_start(self, waveforms):
        return np.full((self._nchan, self._npix), self.window_start,
                       dtype=np.intp)


class PeakFindingIntegrator(WindowIntegrator):
    name = 'PeakFindingIntegrator'
    window_shift = Int(3, help='Define the shift of the integration window '
                               'from the peakpos '
                               '(peakpos - shift)').tag(config=True)
    sig_amp_cut_HG = Int(None, allow_none=True,
                         help='Define the cut above which a sample is '
                              'considered as significant for PeakFinding '
                              'in the HG channel').tag(config=True)
    sig_amp_cut_LG = Int(None, allow_none=True,
                         help='Define the cut above which a sample is '
                              'considered as significant for PeakFinding '
                              'in the LG channel').tag(config=True)

    def __init__(self, config, tool, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)
        self._sig_channel = None
        self._sig_pixels = None
        self.input_shift = self.window_shift

    # Extract significant entries
    def _extract_significant_entries(self, waveforms):
        """
        Obtain the samples that the user has specified as significant.

        Parameters
        ----------
        waveforms : ndarray
            Waveforms stored in a numpy array of shape
            (n_chan, n_pix, n_samples).

        Returns
        -------
        significant_samples : masked ndarray
            Identical to waveforms, except the insignificant samples are
            masked.

        """
        sig_entries = np.ones(waveforms.shape, dtype=bool)
        if self.sig_amp_cut_HG:
            sig_entries[0] = waveforms[0] > self.sig_amp_cut_HG
        if self._nchan > 1 and self.sig_amp_cut_LG:
            sig_entries[1] = waveforms[1] > self.sig_amp_cut_LG
        self._sig_pixels = np.any(sig_entries, axis=2)
        self._sig_channel = np.any(self._sig_pixels, axis=1)
        if not self._sig_channel[0]:
            self.log.error("sigamp value excludes all values in HG channel")
        return np.ma.array(waveforms, mask=~sig_entries)

    def _get_window_start(self, waveforms):
        significant_samples = waveforms
        if self.sig_amp_cut_HG or self.sig_amp_cut_HG:
            significant_samples = self._extract_significant_entries(waveforms)
        else:
            self._sig_channel = np.ones(self._nchan, dtype=bool)
            self._sig_pixels = np.ones((self._nchan, self._npix), dtype=bool)
        self.peakpos = self._find_peak(significant_samples)
        return np.full((self._nchan, self._npix),
                       self.peakpos - self.window_shift,
                       dtype=np.intp)

    @abstractmethod
    def _find_peak(self, significant_samples):
        """
        Find the peak to define integration window around.

        Parameters
        ----------
        significant_samples : masked ndarray
            Numpy array with the significant samples unmasked. Has shape of
            (n_chan, n_pix, n_samples).

        Returns
        -------
        peakpos : ndarray
            Numpy array of the peak position for each pixel. Has shape of
            (n_chan, n_pix).

        """


class GlobalPeakIntegrator(PeakFindingIntegrator):
    name = 'GlobalPeakIntegrator'

    def __init__(self, config, tool, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)

    def _find_peak(self, significant_samples):
        max_t = significant_samples.argmax(2)
        max_s = significant_samples.max(2)

        peakpos = np.zeros((self._nchan, self._npix), dtype=np.int)
        peakpos[0, :] = np.round(np.average(max_t[0], weights=max_s[0]))
        if self._nchan > 1:
            if self._sig_channel[1]:
                peakpos[1, :] = np.round(
                    np.average(max_t[1], weights=max_s[1]))
            else:
                self.log.info("LG not significant, using HG for peak finding "
                              "instead")
                peakpos[1, :] = peakpos[0]
        return peakpos


class LocalPeakIntegrator(PeakFindingIntegrator):
    name = 'LocalPeakIntegrator'

    def __init__(self, config, tool, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)

    def _find_peak(self, significant_samples):
        peakpos = np.full((self._nchan, self._npix),
                          significant_samples.argmax(2),
                          dtype=np.int)
        sig_pix = self._sig_pixels
        if self._nchan > 1:  # If the LG is not significant, use the HG peakpos
            peakpos[1] = np.where(sig_pix[1] < sig_pix[0],
                                  peakpos[0], peakpos[1])
        return peakpos


class NeighbourPeakIntegrator(PeakFindingIntegrator):
    name = 'NeighbourPeakIntegrator'
    lwt = Int(0, help='Weight of the local pixel (0: peak from neighbours '
                      'only, 1: local pixel counts as much '
                      'as any neighbour').tag(config=True)

    def __init__(self, config, tool, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)

    @staticmethod
    def requires_neighbours():
        return True

    def _find_peak(self, significant_samples):
        sig_sam = significant_samples
        max_num_nei = len(max(self.neighbours, key=len))
        allvals = np.zeros((self._nchan, self._npix,
                            max_num_nei + 1, self._nsamples))
        for ipix, neighbours in enumerate(self.neighbours):
            num_nei = len(neighbours)
            allvals[:, ipix, :num_nei, :] = sig_sam[:, neighbours]
            allvals[:, ipix, num_nei, :] = sig_sam[:, ipix] * self.lwt
        sum_data = allvals.sum(2)
        return np.full((self._nchan, self._npix), sum_data.argmax(2),
                       dtype=np.int)


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
