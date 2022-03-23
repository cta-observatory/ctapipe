from ctapipe.core import Component
from ctapipe.containers import ReconstructedGeometryContainer

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

    def __init__(self, *args, **kwargs):
        """
        Create a new instance of ImPACTReconstructor
        """
        super().__init__(*args, **kwargs)

    def predict(self, tels_dict):
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
