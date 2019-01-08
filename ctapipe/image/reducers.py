"""
Algorithms for the data volume reduction.
"""

from abc import abstractmethod

from ctapipe.core import Component
from ctapipe.core.factory import child_subclasses, has_traits
from traitlets import CaselessStrEnum


class DataVolumeReducer(Component):
    """
    Base component for data volume reducers.

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

    def __init__(self, config=None, tool=None, **kwargs):
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
        particular reducer.

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

data_volume_reducers = child_subclasses(DataVolumeReducer)
data_volume_reducer_names = [cls.__name__ for cls in data_volume_reducers]
all_classes = [DataVolumeReducer] + data_volume_reducers
classes_with_traits = [cls for cls in all_classes if has_traits(cls)]
__all__ = data_volume_reducer_names


def enum_trait():
    return CaselessStrEnum(
        data_volume_reducer_names,
        'DataVolumeReducer',
        allow_none=True,
        help=''
    ).tag(config=True)


def from_name(data_volume_reducer_name=None, *args, **kwargs):
    if data_volume_reducer_name is None:
        data_volume_reducer_name = 'DataVolumeReducer'

    cls = globals()[data_volume_reducer_name]
    return cls(*args, **kwargs)
