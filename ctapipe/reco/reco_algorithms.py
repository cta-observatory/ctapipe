from ctapipe.core import Component
from ctapipe.io.containers import ReconstructedShowerContainer

__all__ = ['Reconstructor', 'TooFewTelescopesException', 'InvalidWidthException']


class TooFewTelescopesException(Exception):
    pass


class InvalidWidthException(Exception):
    pass


class Reconstructor(Component):
    """This is the base class from which all direction reconstruction
algorithms should inherit from"""

    def predict(self, tels_dict):
        """overwrite this method with your favourite direction reconstruction
        algorithm

        Parameters
        ----------
        tels_dict : dict
            general dictionary containing all triggered telescopes data

        Returns
        -------
        Standard  `RecoShowerGeom` container

        """
        return ReconstructedShowerContainer()
