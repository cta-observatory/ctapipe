"""
Extraction algorithms to compute the statistics from a sequence of images
"""

__all__ = [
    "StatisticsExtractor",
    "PlainExtractor",
]


from abc import abstractmethod

import numpy as np
import scipy.stats
from traitlets import Int

from ctapipe.core import TelescopeComponent
from ctapipe.containers import StatisticsContainer


class StatisticsExtractor(TelescopeComponent):
    def __init__(self, subarray, config=None, parent=None, **kwargs):
        """
        Base component to handle the extraction of the statistics
        from a sequence of charges and pulse times (images).
>>>>>>> 58d868c8 (added stats extractor parent component)

        Parameters
        ----------
        kwargs
        """
        super().__init__(subarray=subarray, config=config, parent=parent, **kwargs)

    @abstractmethod
    def __call__(self, images, trigger_times) -> list:
        """
        Call the relevant functions to extract the statistics
        for the particular extractor.

        Parameters
        ----------
        images : ndarray
            images stored in a numpy array of shape
            (n_images, n_channels, n_pix).
        trigger_times : ndarray
            images stored in a numpy array of shape
            (n_images, )

        Returns
        -------
        List StatisticsContainer:

            List of extracted statistics and validity ranges
        """

class PlainExtractor(StatisticsExtractor):
    """
    Extractor the statistics from a sequence of images
    using numpy and scipy functions
    """

    sample_size = Int(2500, help="sample size").tag(config=True)

    def __call__(self, dl1_table, col_name="image") -> list:
       
        # in python 3.12 itertools.batched can be used
        image_chunks = (dl1_table[col_name].data[i:i + self.sample_size] for i in range(0, len(dl1_table[col_name].data), self.sample_size))
        time_chunks = (dl1_table["time"][i:i + self.sample_size] for i in range(0, len(dl1_table["time"]), self.sample_size))

        # Calculate the statistics from a sequence of images
        stats_list = []
        for img, time in zip(image_chunks,time_chunks):
            stats_list.append(self._plain_extraction(img, time))
            
        return stats_list

    def _plain_extraction(self, images, trigger_times) -> StatisticsContainer:
        return StatisticsContainer(
            validity_start=trigger_times[0],
            validity_stop=trigger_times[-1],
            max=np.max(images, axis=0),
            min=np.min(images, axis=0),
            mean=np.nanmean(images, axis=0),
            median=np.nanmedian(images, axis=0),
            std=np.nanstd(images, axis=0),
            skewness=scipy.stats.skew(images, axis=0),
            kurtosis=scipy.stats.kurtosis(images, axis=0),
        )
