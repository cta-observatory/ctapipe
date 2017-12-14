"""
Algorithms for the data volume reduction.
"""

from abc import abstractmethod
from traitlets import CaselessStrEnum
from ctapipe.core import Component, Factory

__all__ = ['DataVolumeReductor', 'DataVolumeReductorFactory']


class DataVolumeReductor(Component):
    """
    Base component for data volume reductors.

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
    def __init__(self, config, tool, **kwargs):
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
    def reduce_waveforms(self, waveforms):
        """
        Call the relevant functions to reduce the waveforms using a
        particular reductor.

        Parameters
        ----------
        waveforms : ndarray
            Waveforms stored in a numpy array of shape
            (n_chan, n_pix, n_samples).

        Returns
        -------
        reduced_waveforms : ndarray
            Reduced waveforms stored in a numpy array of shape
            (n_chan, n_pix, n_samples).
        """


class DataVolumeReductorFactory(Factory):
    """
    Factory class for creating a DataVolumeReductor
    """
    description = "Obtain DataVolumeReductor based on reductor traitlet"

    subclasses = Factory.child_subclasses(DataVolumeReductor)
    subclass_names = [c.__name__ for c in subclasses]

    reductor = CaselessStrEnum(subclass_names, 'NeighbourPeakIntegrator',
                               help='Data volume reduction scheme to use for '
                                    'the conversion to dl0.').tag(config=True)

    # Product classes traits

    def get_factory_name(self):
        return self.__class__.__name__

    def get_product_name(self):
        return self.reductor
