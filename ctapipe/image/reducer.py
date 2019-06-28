"""
Algorithms for the data volume reduction.
"""

from abc import abstractmethod

from ctapipe.core import Component

__all__ = [
    'DataVolumeReducer',
    'NullDataVolumeReducer',
]


class DataVolumeReducer(Component):
    """
    Base component for data volume reducers.
    """

    @abstractmethod
    def __call__(self, waveforms):
        """
        Call the relevant functions to perform data volume reduction on the
        waveforms.

        Parameters
        ----------
        waveforms : ndarray
            Waveforms stored in a numpy array of shape
            (n_pix, n_samples).

        Returns
        -------
        reduced_waveforms : ndarray
            Reduced waveforms stored in a numpy array of shape
            (n_pix, n_samples).
        """


class NullDataVolumeReducer(DataVolumeReducer):
    """
    Perform no data volume reduction
    """

    def __call__(self, waveforms):
        return waveforms
