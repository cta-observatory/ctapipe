from ctapipe.core import Component, QualityQuery
from ctapipe.containers import ReconstructedGeometryContainer, ArrayEventContainer
from abc import abstractmethod

from ctapipe.core.traits import List

__all__ = ["Reconstructor", "TooFewTelescopesException", "InvalidWidthException"]


class TooFewTelescopesException(Exception):
    pass


class InvalidWidthException(Exception):
    pass


class Reconstructor(Component):
    """
    This is the base class from which all direction reconstruction
    algorithms should inherit from
    """

    def __init__(self, subarray, **kwargs):
        super().__init__(**kwargs)
        self.subarray = subarray

    @abstractmethod
    def __call__(self, event: ArrayEventContainer):
        """overwrite this method with your favourite direction reconstruction
        algorithm

        Parameters
        ----------
        tels_dict : dict
            general dictionary containing all triggered telescopes data

        Returns
        -------
        `~ctapipe.containers.ReconstructedGeometryContainer`

        """
        return ReconstructedGeometryContainer()


class ShowerQualityQuery(QualityQuery):
    """Configuring shower-wise data checks."""

    quality_criteria = List(
        default_value=[
            ("> 50 phe", "lambda p: p.hillas.intensity > 50"),
            ("Positive width", "lambda p: p.hillas.width.value > 0"),
            ("> 3 pixels", "lambda p: p.morphology.num_pixels > 3"),
        ],
        help=QualityQuery.quality_criteria.help,
    ).tag(config=True)
