"""
Charge extraction algorithms to reduce the image to one value per pixel
"""

__all__ = ['ChargeExtractorFactory', 'FullIntegrator', 'SimpleIntegrator',
           'GlobalPeakIntegrator', 'LocalPeakIntegrator',
           'NeighbourPeakIntegrator', 'AverageWfPeakIntegrator']


from abc import abstractmethod
import numpy as np
from traitlets import Int, CaselessStrEnum, Float
from ctapipe.core import Component, Factory
from ctapipe.utils.neighbour_sum import get_sum_array

class FullWaveAxes:
    channel = 0
    pixel = 1
    sample = 2

class WaveAxes:
    pixel = 0
    sample = 1


class ChargeExtractor(Component):

    def __init__(self, config=None, tool=None, **kwargs):
        """
        Base component to handle the extraction of charge from an image cube.

        Attributes
        ----------
        neighbours : ndarray
            2D array where each row is [pixel index, one neighbour
            of that pixel].
            Changes per telescope.
            Can be obtained from
            `ctapipe.instrument.CameraGeometry.neighbor_matrix_where`.

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
    def get_peakpos(self, waveforms):
        """
        Get the peak position from the waveforms

        Parameters
        ----------
        waveforms : ndarray
            Waveforms stored in a numpy array of shape
            (n_chan, n_pix, n_samples).

        Returns
        -------
        peakpos : ndarray
            Numpy array of the peak position for each pixel. 
            Has shape of (n_chan, n_pix).

        """

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
        window : ndarray
            Bool numpy array defining the samples included in the integration
            window.
        """


class Integrator(ChargeExtractor):

    def __init__(self, config=None, tool=None, **kwargs):
        """
        Base component for charge extractors that perform integration.

        Attributes
        ----------
        neighbours : list
            List of neighbours for each pixel. Changes per telescope.

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

    @staticmethod
    def check_window_width_and_start(n_samples, start, width):
        """
        Check that the combination of window width and start positions fit
        within the readout window.

        Parameters
        ----------
        n_samples : int
            Number of samples in the waveform
        start : ndarray
            Numpy array containing the window start for each pixel. Shape =
            (n_chan, n_pix)
        width : ndarray
            Numpy array containing the window width for each pixel. Shape =
            (n_chan, n_pix)

        """
        width[width > n_samples] = n_samples
        start[start < 0] = 0
        sum_check = start + width > n_samples
        start[sum_check] = n_samples - width[sum_check]

    @abstractmethod
    def get_peakpos(self, waveforms):
        """"""

    @abstractmethod
    def _get_window_start(self, waveforms, peakpos):
        """
        Get the starting point for the integration window

        Parameters
        ----------
        waveforms : ndarray
            Waveforms stored in a numpy array of shape
            (n_chan, n_pix, n_samples).
        peakpos : ndarray
            Numpy array containing the peak position for each pixel. 
            Shape = (n_chan, n_pix)

        Returns
        -------
        start : ndarray
            Numpy array containing the window start for each pixel. 
            Shape = (n_chan, n_pix)

        """

    @abstractmethod
    def _get_window_width(self, waveforms):
        """
        Get the width of the integration window

        Parameters
        ----------
        waveforms : ndarray
            Waveforms stored in a numpy array of shape
            (n_chan, n_pix, n_samples).

        Returns
        -------
        width : ndarray
            Numpy array containing the window width for each pixel. 
            Shape = (n_chan, n_pix)

        """

    def get_start_and_width(self, waveforms, peakpos):
        """
        Obtain the numpy arrays containing the start and width for the
        integration window for each pixel.

        Parameters
        ----------
        waveforms : ndarray
            Waveforms stored in a numpy array of shape
            (n_chan, n_pix, n_samples).
        peakpos : ndarray
            Numpy array of the peak position for each pixel. 
            Has shape of (n_chan, n_pix).

        Returns
        -------
        w_start : ndarray
            Numpy array containing the Start sample of integration window.
            Shape: (n_chan, n_pix).
        w_width : ndarray
            Numpy array containing the window size of integration window.
            Shape (n_chan, n_pix).
        """
        w_start = self._get_window_start(waveforms, peakpos)
        w_width = self._get_window_width(waveforms)
        n_samples = waveforms.shape[WaveAxes.sample]
        self.check_window_width_and_start(n_samples, w_start, w_width)
        return w_start, w_width

    @staticmethod
    def get_window(waveforms, start, width):
        """
        Build the a numpy array of bools defining the integration window.

        Parameters
        ----------
        waveforms : ndarray
            Waveforms stored in a numpy array of shape
            (n_chan, n_pix, n_samples).
        start : ndarray
            Numpy array containing the Start sample of integration window.
            Shape: (n_chan, n_pix).
        width : ndarray
            Numpy array containing the window size of integration window.
            Shape (n_chan, n_pix).

        Returns
        -------
        integration_window : ndarray
            Numpy array containing True where the samples lay within the
            integration window, and False where the samples lay outside. Has
            shape of (n_chan, n_pix, n_samples).

        """
        end = start + width

        # Obtain integration window using the indices of the waveforms array
        ind = np.indices(waveforms.shape)[WaveAxes.sample]
        integration_window = (ind >= start[..., None]) & (ind < end[..., None])
        return integration_window

    @staticmethod
    def extract_from_window(waveforms, window):
        """
        Extract the charge but applying an intregration window to the 
        waveforms.

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
        charge : ndarray
            Extracted charge stored in a numpy array of shape (n_chan, n_pix).
        """
        windowed = np.ma.array(waveforms, mask=~window)
        charge = windowed.sum(axis=WaveAxes.sample).data
        return charge

    def get_window_from_waveforms(self, waveforms):
        """
        Consolidating function to obtain the window and peakpos given 
        a waveform.

        Parameters
        ----------
        waveforms : ndarray
            Waveforms stored in a numpy array of shape
            (n_chan, n_pix, n_samples).

        Returns
        -------
        window : ndarray
            Numpy array containing True where the samples lay within the
            integration window, and False where the samples lay outside. Has
            shape of (n_chan, n_pix, n_samples).
        peakpos : ndarray
            Numpy array of the peak position for each pixel. 
            Has shape of (n_chan, n_pix).

        """
        peakpos = self.get_peakpos(waveforms)
        start, width = self.get_start_and_width(waveforms, peakpos)
        window = self.get_window(waveforms, start, width)
        return window, peakpos

    def extract_charge(self, waveforms):
        window, peakpos = self.get_window_from_waveforms(waveforms)
        charge = self.extract_from_window(waveforms, window)
        return charge, peakpos, window


class FullIntegrator(Integrator):

    def __init__(self, config=None, tool=None, **kwargs):
        """
        Charge extractor that integrates the entire waveform.

        Attributes
        ----------
        neighbours : list
            List of neighbours for each pixel. Changes per telescope.

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

    def _get_window_start(self, waveforms, peakpos):
        npix, nsamples = waveforms.shape
        return np.zeros(npix, dtype=np.intp)

    def _get_window_width(self, waveforms):
        npix, nsamples = waveforms.shape
        return np.full(npix, nsamples, dtype=np.intp)

    def get_peakpos(self, waveforms):
        npix, nsamples = waveforms.shape
        return np.zeros(npix, dtype=np.intp)


class WindowIntegrator(Integrator):
    window_shift = Int(3, help='Define the shift of the integration window '
                               'from the peakpos '
                               '(peakpos - shift)').tag(config=True)
    window_width = Int(7, help='Define the width of the integration '
                               'window').tag(config=True)

    def __init__(self, config=None, tool=None, **kwargs):
        """
        Base component for charge extractors that perform integration within
        a window.

        Attributes
        ----------
        neighbours : list
            List of neighbours for each pixel. Changes per telescope.

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

    def get_peakpos(self, waveforms):
        return self._obtain_peak_position(waveforms)

    def _get_window_start(self, waveforms, peakpos):
        return peakpos - self.window_shift

    def _get_window_width(self, waveforms):
        npix, nsamples = waveforms.shape
        return np.full(npix, self.window_width, dtype=np.intp)

    @abstractmethod
    def _obtain_peak_position(self, waveforms):
        """
        Find the peak to define integration window around.

        Parameters
        ----------
        waveforms : ndarray
            Waveforms stored in a numpy array of shape
            (n_chan, n_pix, n_samples).

        Returns
        -------
        peakpos : ndarray
            Numpy array of the peak position for each pixel. Has shape of
            (n_chan, n_pix).

        """


class SimpleIntegrator(WindowIntegrator):
    t0 = Int(0, help='Define the peak position for all '
                     'pixels').tag(config=True)

    def __init__(self, config=None, tool=None, **kwargs):
        """
        Charge extractor that integrates within a window defined by the user.

        Attributes
        ----------
        neighbours : list
            List of neighbours for each pixel. Changes per telescope.

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

    def _obtain_peak_position(self, waveforms):
        npix, nsamples = waveforms.shape
        return np.full(npix, self.t0, dtype=np.intp)


class PeakFindingIntegrator(WindowIntegrator):
    peak_detection_threshold = Float(None, allow_none=True,
                                     help='Define the cut above which a sample is '
                                'considered as significant for PeakFinding '
                                'in the HG channel').tag(config=True)

    def __init__(self, config=None, tool=None, **kwargs):
        """
        Base component for charge extractors that perform integration within
        a window defined around a peak position.

        Attributes
        ----------
        neighbours : list
            List of neighbours for each pixel. Changes per telescope.

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
        self._sig_pixels = None

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
        npix, nsamples = waveforms.shape
        if self.peak_detection_threshold:
            sig_entries = np.ones(waveforms.shape, dtype=bool)
            if self.peak_detection_threshold:
                sig_entries = waveforms >  self.peak_detection_threshold
            self._sig_pixels = np.any(sig_entries, axis=WaveAxes.sample)
            return np.ma.array(waveforms, mask=~sig_entries)
        else:
            self._sig_pixels = np.ones(npix, dtype=bool)
            return waveforms

    @abstractmethod
    def _obtain_peak_position(self, waveforms):
        """"""


class GlobalPeakIntegrator(PeakFindingIntegrator):
    """
    Charge extractor that defines an integration window about the global
    peak in the image.

    Attributes
    ----------
    neighbours : list
        List of neighbours for each pixel. Changes per telescope.

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

    def __init__(self, config=None, tool=None, **kwargs):

        super().__init__(config=config, tool=tool, **kwargs)

    def _obtain_peak_position(self, waveforms):
        npix, nsamples = waveforms.shape
        significant_samples = self._extract_significant_entries(waveforms)
        max_t = significant_samples.argmax(axis=WaveAxes.sample)
        max_s = significant_samples.max(axis=WaveAxes.sample)

        peakpos = np.zeros(npix, dtype=np.int)
        peakpos[:] = np.round(np.average(max_t, weights=max_s))

        return peakpos


class LocalPeakIntegrator(PeakFindingIntegrator):
    """
     Charge extractor that defines an integration window about the local
     peak in each pixel.

     Attributes
     ----------
     neighbours : list
         List of neighbours for each pixel. Changes per telescope.

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

    def __init__(self, config=None, tool=None, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)

    def _obtain_peak_position(self, waveforms):
        npix, nsamples = waveforms.shape
        significant_samples = self._extract_significant_entries(waveforms)
        peakpos = np.full(
            npix,
            significant_samples.argmax(WaveAxis.samples),
            dtype=np.int
        )
        sig_pix = self._sig_pixels

        return peakpos


class NeighbourPeakIntegrator(PeakFindingIntegrator):
    """
    Charge extractor that defines an integration window defined by the 
    peaks in the neighbouring pixels.

    Attributes
    ----------
    neighbours : list
        List of neighbours for each pixel. Changes per telescope.

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
    lwt = Int(0, help='Weight of the local pixel (0: peak from neighbours '
                      'only, 1: local pixel counts as much '
                      'as any neighbour').tag(config=True)

    def __init__(self, config=None, tool=None, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)

    @staticmethod
    def requires_neighbours():
        return True

    def _obtain_peak_position(self, waveforms):
        npix, nsamp = waveforms.shape
        nchan = 1
        significant_samples = self._extract_significant_entries(waveforms)
        sig_sam = significant_samples.astype(np.float32)
        sum_data = np.zeros_like(sig_sam)
        neighbors = self.neighbours.astype(np.uint16)
        neighbors_length = neighbors.shape[0]
        get_sum_array(sig_sam, sum_data,
                      nchan, npix, nsamp,
                      neighbors, neighbors_length, self.lwt)
        return sum_data.argmax(WaveAxes.sample).astype(np.int)


class AverageWfPeakIntegrator(PeakFindingIntegrator):
    """
    Charge extractor that defines an integration window defined by the 
    average waveform across all pixels.

    Attributes
    ----------
    neighbours : list
        List of neighbours for each pixel. Changes per telescope.

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

    def __init__(self, config=None, tool=None, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)

    def _obtain_peak_position(self, waveforms):
        npix, nsamples = waveforms.shape
        significant_samples = self._extract_significant_entries(waveforms)
        peakpos = np.zeros(npix, dtype=np.int)
        avg_wf = np.mean(significant_samples, axis=WaveAxes.pixel)
        peakpos += np.argmax(avg_wf, axis=WaveAxes.pixel)[:, None]
        return peakpos


class ChargeExtractorFactory(Factory):
    """
    Factory to obtain a ChargeExtractor.
    """
    base = ChargeExtractor
    default = 'NeighbourPeakIntegrator'
    custom_product_help = 'Charge extraction scheme to use.'
